import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from wrapper import UnityGymnasiumEnv
from reward_calc import compute_reward_scaled


class PythonRewardWrapper(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, unity_file=None, no_graphics=True, worker_id=0, timeout=300):
        self.env = UnityGymnasiumEnv(
            file_name=unity_file, no_graphics=no_graphics, worker_id=worker_id, timeout=timeout
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.prev_obs = None
        self.episode_reward = 0
        self.episode_length = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        self.episode_reward = 0
        self.episode_length = 0
        return obs, info

    def step(self, action):
        import numpy as _np

        if isinstance(action, (list, tuple)):
            a = _np.asarray(action)
        else:
            a = _np.asarray(action)

        if a.ndim > 1:
            a = a.reshape(-1)

        if hasattr(self.action_space, "nvec") or str(self.action_space).startswith("MultiDiscrete"):
            a = a.astype(_np.int32)
            action_to_send = a.tolist()
        else:
            action_to_send = a.astype(_np.float32).tolist()

        obs, _, terminated, truncated, info = self.env.step(action_to_send)

        reward = compute_reward_scaled(self.prev_obs, action_to_send, obs)

        self.episode_reward += reward
        self.episode_length += 1

        if terminated or truncated:
            info["episode"] = {"r": self.episode_reward, "l": self.episode_length}

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
        for idx, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][idx]
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
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


if __name__ == "__main__":
    unity_file = "C:/Users/TheTr/maturita/builds/FatalPes"
    no_graphics = True
    worker_id = 0

    total_timesteps = 5_000_000  # you already set this; keep it if you want long training

    env = DummyVecEnv([make_env(unity_file, no_graphics, worker_id)])

    # Re-enable VecNormalize BUT do not squish rewards to tiny range:
    # - norm_obs=True helps training stability
    # - norm_reward=True helps stabilize large per-episode magnitudes
    # - clip_reward: set to a large value so +50 / -10 are still visible but extreme outliers are bounded
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=100.0,  # large clipping so goal/crash still matter
        gamma=0.99,
        epsilon=1e-8,
    )

    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    print("\nInitializing PPO model...")

    # Use a linear learning rate schedule (stable-baselines accepts a function)
    initial_lr = 3e-4
    lr_schedule = lambda progress_remaining: initial_lr * progress_remaining  # linear decay

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=lr_schedule,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,  # modest entropy to keep exploration alive (not huge)
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tensorboard_logs/",
    )

    reward_callback = RewardLoggingCallback(verbose=1)

    try:
        model.learn(total_timesteps=total_timesteps, callback=reward_callback)

        print("\nTraining completed!")

        model_path = "ppo_unity_with_vecnorm"
        model.save(model_path)
        # Save VecNormalize stats so you can reload exactly:
        env.save("vecnormalize_stats.pkl")

        print(f"Model saved to: {model_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current progress...")
        model.save("ppo_unity_interrupted")
        env.save("vecnormalize_stats_interrupted.pkl")
        print("Progress saved!")

    finally:
        env.close()
        print("Environment closed.")
