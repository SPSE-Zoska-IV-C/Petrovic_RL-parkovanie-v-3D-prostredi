# train_sac.py
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from wrapper import UnityGymnasiumEnv
from reward_calc import compute_reward_continuous  # <-- use continuous reward


class PythonRewardWrapper(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, unity_file=None, no_graphics=True, worker_id=0, timeout=300):
        self.env = UnityGymnasiumEnv(
            file_name=unity_file, no_graphics=no_graphics, worker_id=worker_id, timeout=timeout
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.prev_obs = None
        self.episode_reward = 0.0
        self.episode_length = 0

    def reset(self, **kwargs):
        # wrapper.reset returns (obs, info)
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        self.episode_reward = 0.0
        self.episode_length = 0
        return obs, info

    def step(self, action):
        import numpy as _np

        a = _np.asarray(action, dtype=_np.float32)
        if a.ndim > 1:
            a = a.reshape(-1)

        # prepare to send to wrapper
        if hasattr(self.action_space, "nvec") or str(self.action_space).startswith("MultiDiscrete"):
            a = a.astype(_np.int32)
            action_to_send = a.tolist()
        else:
            action_to_send = a.astype(_np.float32).tolist()

        # call the underlying wrapper env
        obs, _, terminated, truncated, info = self.env.step(action_to_send)

        # use prev_obs from info if provided by wrapper, else fallback to stored self.prev_obs
        prev_obs = info.get("prev_obs", self.prev_obs)

        # compute reward and pass info so reward fn can use prev_action/truncation
        reward = compute_reward_continuous(prev_obs, action_to_send, obs, info)

        self.episode_reward += reward
        self.episode_length += 1

        if terminated or truncated:
            info["episode"] = {"r": self.episode_reward, "l": self.episode_length}

        # update local prev_obs for next step (backup)
        self.prev_obs = obs
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()


class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for idx, done in enumerate(dones):
            if done:
                info = infos[idx]
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    # log rolling stats
                    self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards[-100:]))
                    self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths[-100:]))
                    self.logger.record("rollout/ep_rew_last", ep_reward)
        return True


def make_env(unity_file, no_graphics, worker_id):
    def _init():
        return PythonRewardWrapper(
            unity_file=unity_file, no_graphics=no_graphics, worker_id=worker_id, timeout=300
        )

    return _init


def sanity_checks_env(env, expected_action_dim=3, expected_obs_dim=18):
    # env is a vectorized env (VecNormalize wraps DummyVecEnv)
    # extract the underlying env action and obs spaces by resetting once
    obs = env.reset()
    a_space = env.action_space
    o_space = env.observation_space

    print(f"[SANITY] action_space: {a_space}")
    print(f"[SANITY] observation_space: {o_space}")
    # action space must be Box for SAC
    if a_space.__class__.__name__.lower().find("box") == -1:
        raise RuntimeError(f"Action space must be Box (continuous) for SAC. Got: {a_space}")

    action_dim = int(np.prod(a_space.shape))
    if action_dim != expected_action_dim:
        raise RuntimeError(
            f"Action dim mismatch: Unity reports {action_dim}, expected {expected_action_dim}. Fix BehaviorParameters in Unity."
        )

    obs_dim = int(np.prod(o_space.shape))
    print(f"[SANITY] observation dim detected: {obs_dim} (expected ~{expected_obs_dim})")
    if obs_dim != expected_obs_dim:
        print(
            f"[SANITY WARNING] Observation dim ({obs_dim}) != expected ({expected_obs_dim}). If intentional, adjust expected_obs_dim in this script."
        )


if __name__ == "__main__":
    unity_file = "C:/Users/TheTr/maturita/builds/ActionEatest"  # edit to match your path or None
    no_graphics = True
    worker_id = 0

    total_timesteps = 500_000  # SAC needs many samples; adjust

    env = DummyVecEnv([make_env(unity_file, no_graphics, worker_id)])

    # VecNormalize for obs only (norm_reward must be False for off-policy algorithms)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)

    # Sanity checks: fail fast if Unity/Python mismatch
    sanity_checks_env(env, expected_action_dim=3, expected_obs_dim=18)

    # print("Creating SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device="auto",
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        tensorboard_log="./tensorboard_sac/",
    )

    callback = RewardLoggingCallback(verbose=1)

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=1)
        print("Training completed.")
        model.save("sac_unity_model")
        env.save("vecnormalize_sac.pkl")
    except KeyboardInterrupt:
        print("Interrupted by user. Saving partial progress...")
        model.save("sac_unity_interrupted")
        env.save("vecnormalize_sac_interrupted.pkl")
    finally:
        env.close()
        print("Env closed.")
