# easy env - here are the parameters for eazy env model, training script
import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from wrapper_eazy import UnityGymnasiumEnv
from reward_calc import compute_reward_continuous


class PythonRewardWrapper(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, unity_file=None, no_graphics=True, worker_id=0, timeout=300):
        self.env = UnityGymnasiumEnv(
            file_name=unity_file,
            no_graphics=no_graphics,
            worker_id=worker_id,
            timeout=timeout
        )
    
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.prev_obs = None
        self.episode_reward = 0.0
        self.episode_length = 0
        self.crash_handled = False

    def reset(self, **kwargs):
        reset_out = self.env.reset(**kwargs)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            obs = reset_out
            info = {}
        self.prev_obs = obs
        self.episode_reward = 0.0
        self.episode_length = 0
        self.crash_handled = False
        return obs, info

    def step(self, action):
        import numpy as _np

        a = _np.asarray(action, dtype=_np.float32)
        if a.ndim > 1:
            a = a.reshape(-1)

        if hasattr(self.action_space, "nvec") or str(self.action_space).startswith("MultiDiscrete"):
            a = a.astype(_np.int32)
            action_to_send = a.tolist()
        else:
            action_to_send = a.astype(_np.float32).tolist()

        out = self.env.step(action_to_send)
        if len(out) == 5:
            obs, _, terminated, truncated, info = out
        elif len(out) == 4:
            obs, _, done, info = out
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError("Unexpected env.step() return length: %d" % len(out))

        info['terminated'] = bool(terminated)
        info['truncated'] = bool(truncated)
        if (terminated or truncated) and 'terminal_observation' not in info:
            try:
                info['terminal_observation'] = np.asarray(obs).tolist()
            except Exception:
                info['terminal_observation'] = None
        if truncated:
            info['TimeLimit.truncated'] = True


        try:
            _obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
            total_len = _obs_arr.shape[0]
            idx_maybe_crash = total_len - 2 if total_len >= 2 else None
            if idx_maybe_crash is not None:
                maybe_val = float(_obs_arr[idx_maybe_crash])
                if maybe_val >= 0.5:
                    info['crash'] = True
                    info['obs_crash_val'] = maybe_val
                else:
                    info.setdefault('crash', False)
                    info.setdefault('obs_crash_val', maybe_val)
        except Exception:
            info.setdefault('crash', info.get('crash', False))
        # -------------------------------------------------------------------------

        prev_obs = info.get("prev_obs", self.prev_obs)
        info['crash_handled'] = bool(self.crash_handled)

        reward = compute_reward_continuous(prev_obs, action_to_send, obs, info)

        self.crash_handled = bool(info.get('crash_handled', self.crash_handled))

        try:
            if isinstance(reward, (list, tuple, np.ndarray)):
                reward = float(np.sum(reward))
            else:
                reward = float(reward)
        except Exception:
            reward = 0.0

        if info.get('zero_episode'):
            self.episode_length += 1
        else:
            self.episode_reward += reward
            self.episode_length += 1

        if terminated or truncated:
            if info.get('zero_episode'):
                info['episode'] = {'r': 0.0, 'l': self.episode_length}
                self.episode_reward = 0.0
            else:
                info['episode'] = {'r': float(self.episode_reward), 'l': self.episode_length}

            if 'termination_reason' not in info:
                if info.get('TimeLimit.truncated', False) or truncated:
                    info['termination_reason'] = 'timeout'
                elif info.get('goal_reached', False):
                    info['termination_reason'] = 'goal'
                else:
                    info['termination_reason'] = 'terminated'

        self.prev_obs = obs
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()


class RewardLoggingCallback(BaseCallback):
    def __init__(self, print_interval=100000, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.print_interval = print_interval

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        for idx, done in enumerate(dones):
            if done:
                info = infos[idx]
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                else:
                    ep_reward = float('nan')
                    ep_length = -1

                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                termination_type = 'unknown'
                if info.get('TimeLimit.truncated', False) or info.get('truncated', False):
                    termination_type = 'truncated'
                elif info.get('is_success', False) or info.get('success', False) or info.get('goal_reached', False) or info.get('achieved_goal', False):
                    termination_type = 'goal'
                elif not np.isnan(ep_reward) and ep_reward >= 49.0:
                    termination_type = 'goal'
                elif info.get('terminal_observation') is not None:
                    termination_type = 'terminated'

                if len(self.episode_rewards) > 0:
                    self.logger.record('rollout/ep_rew_mean', np.mean(self.episode_rewards[-100:]))
                    self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths[-100:]))
                self.logger.record('rollout/ep_rew_last', ep_reward)

                ep_idx = len(self.episode_rewards)
                print(f"[EP END #{ep_idx}] reason={termination_type} reward={ep_reward:.2f} length={ep_length}")

                if ep_idx % self.print_interval == 0:
                    print(f"[LOG] Episode {ep_idx}: mean_reward={np.mean(self.episode_rewards[-100:]):.2f}, mean_length={np.mean(self.episode_lengths[-100:]):.2f}, last_reward={ep_reward:.2f}")
        return True


def make_env(unity_file, no_graphics, worker_id, timeout=300):
    def _init():
        return PythonRewardWrapper(unity_file=unity_file, no_graphics=no_graphics, worker_id=worker_id, timeout=timeout)
    return _init


def sanity_checks_env(env, expected_action_dim=2, expected_obs_dim=19):
    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs = reset_out

    if isinstance(obs, np.ndarray) and obs.ndim > 1:
        sample_obs = obs[0]
    else:
        sample_obs = obs

    a_space = env.action_space
    o_space = env.observation_space

    print(f"[SANITY] action_space: {a_space}")
    print(f"[SANITY] observation_space: {o_space}")
    if a_space.__class__.__name__.lower().find("box") == -1:
        raise RuntimeError(f"Action space must be Box (continuous) for SAC. Got: {a_space}")

    action_dim = int(np.prod(a_space.shape))
    if action_dim != expected_action_dim:
        raise RuntimeError(f"Action dim mismatch: Unity reports {action_dim}, expected {expected_action_dim}. Fix BehaviorParameters in Unity.")

    obs_dim = int(np.prod(o_space.shape))
    print(f"[SANITY] observation dim detected: {obs_dim} (expected ~{expected_obs_dim})")
    if obs_dim != expected_obs_dim:
        print(f"[SANITY WARNING] Observation dim ({obs_dim}) != expected ({expected_obs_dim}). If intentional, adjust expected_obs_dim in this script.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--unity-file", type=str, default="EazyFix.exe", help="Path to Unity executable.")
    p.add_argument("--no-graphics", action="store_true", default=True, help="Run Unity builds in no-graphics mode.")
    p.add_argument("--n-envs", type=int, default=8, help="Number of parallel Unity envs.")
    p.add_argument("--worker-offset", type=int, default=0, help="Offset for worker_id to avoid port conflicts.")
    p.add_argument("--total-timesteps", type=int, default=2_500_000, help="Timesteps to learn.")
    return p.parse_args()


def main():
    args = parse_args()
    unity_file = args.unity_file
    no_graphics = args.no_graphics
    n_envs = args.n_envs
    worker_offset = args.worker_offset
    total_timesteps = args.total_timesteps

    print(f"[MAIN] Creating {n_envs} envs from '{unity_file}' (worker_offset={worker_offset})")
    env_fns = [make_env(unity_file, no_graphics, worker_id=(worker_offset + i)) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    sanity_checks_env(env, expected_action_dim=2, expected_obs_dim=19)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device="auto",
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        tensorboard_log="./tensorboard_sac_parallel/",
    )

    callback = RewardLoggingCallback(print_interval=100000, verbose=1)

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=1)
        print("Training completed.")
        model.save("sacE52")
        env.save("sacE52.pkl")
    except KeyboardInterrupt:
        print("Interrupted by user. Saving partial progress...")
        model.save("sac_unity_interrupted_parallel")
        env.save("vecnormalize_sac_interrupted_parallel.pkl")
    finally:
        env.close()
        print("Env closed.")


if __name__ == "__main__":
    main()
