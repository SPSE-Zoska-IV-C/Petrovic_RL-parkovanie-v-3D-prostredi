# train_hard.py
import os
import argparse
import numpy as np
import gymnasium as gym
 
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
 
from wrapper_h import UnityGymnasiumEnv
from reeward_hardo import compute_reward_continuous
 
 
class PythonRewardWrapper(gym.Env):
    metadata = {"render.modes": []}
 
    def __init__(self, unity_file=None, no_graphics=True, worker_id=0, timeout=30):
        self.env = UnityGymnasiumEnv(
            file_name=unity_file,
            no_graphics=no_graphics,
            worker_id=worker_id,
            timeout=timeout,
        )
        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space
        self._reset_episode_state()
 
    def _reset_episode_state(self):
        self.prev_obs              = None
        self.episode_reward_so_far = 0.0
        self.episode_length        = 0
        self.milestones_hit        = set()
 
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs, info = out if (isinstance(out, tuple) and len(out) == 2) else (out, {})
        self._reset_episode_state()
        self.prev_obs = obs
        return obs, info
 
    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if hasattr(self.action_space, "nvec") or str(self.action_space).startswith("MultiDiscrete"):
            action_to_send = a.astype(np.int32).tolist()
        else:
            action_to_send = a.astype(np.float32).tolist()
 
        out = self.env.step(action_to_send)
        if len(out) == 5:
            obs, _, terminated, truncated, info = out
        elif len(out) == 4:
            obs, _, done, info = out
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError(f"Unexpected env.step() return length: {len(out)}")
 
        info["terminated"] = bool(terminated)
        info["truncated"]  = bool(truncated)
 
        if truncated:
            info["TimeLimit.truncated"] = True
 
        info["_raw_obs"]               = np.asarray(obs, dtype=np.float32).copy()
        info["_episode_reward_so_far"] = self.episode_reward_so_far
        info["_milestones_hit"]        = self.milestones_hit
 
        reward = compute_reward_continuous(self.prev_obs, action_to_send, obs, info)
 
        try:
            reward = float(np.sum(reward)) if isinstance(reward, (list, tuple, np.ndarray)) else float(reward)
        except Exception:
            reward = 0.0
 
        self.milestones_hit = info.get("_milestones_hit", self.milestones_hit)
 
        is_terminal = bool(terminated or truncated)
 
        if not is_terminal:
            self.episode_reward_so_far += reward
 
        self.episode_length += 1
 
        if is_terminal:
            total_ep_reward = self.episode_reward_so_far + reward
            info["episode"] = {"r": float(total_ep_reward), "l": self.episode_length}
 
            if "termination_reason" not in info:
                if info.get("crash", False) or info.get("crashed", False):
                    info["termination_reason"] = "crash"
                elif info.get("goal", False) or info.get("goal_reached", False):
                    info["termination_reason"] = "goal"
                else:
                    info["termination_reason"] = "timeout"
 
        self.prev_obs = obs
        return obs, reward, terminated, truncated, info
 
    def close(self):
        self.env.close()
 
 
class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._counts = {"goal": 0, "crash": 0, "timeout": 0, "UNKNOWN": 0}
 
    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if not done:
                continue
 
            ep_reward = info["episode"]["r"] if "episode" in info else float("nan")
            ep_length = info["episode"]["l"] if "episode" in info else -1
            reason    = info.get("termination_reason", "UNKNOWN")
 
            if reason not in ("goal", "crash", "timeout"):
                print(f"[CALLBACK WARNING] unexpected termination_reason='{reason}'")
                reason = "UNKNOWN"
 
            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)
            self._counts[reason] = self._counts.get(reason, 0) + 1
            ep_idx = len(self.episode_rewards)
 
            print(
                f"[EP #{ep_idx}] reward={ep_reward:.2f} len={ep_length} reason={reason} | "
                f"goals={self._counts['goal']} crashes={self._counts['crash']} "
                f"timeouts={self._counts['timeout']}"
            )
        return True
 
 
def make_env(unity_file, no_graphics, worker_id, timeout=30):
    def _init():
        return PythonRewardWrapper(
            unity_file=unity_file,
            no_graphics=no_graphics,
            worker_id=worker_id,
            timeout=timeout,
        )
    return _init
 
 
def sanity_checks_env(env, expected_action_dim=2, expected_obs_dim=19):
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    print(f"[SANITY] action_space:      {env.action_space}")
    print(f"[SANITY] observation_space: {env.observation_space}")
    if "box" not in env.action_space.__class__.__name__.lower():
        raise RuntimeError(f"Action space must be Box for SAC. Got: {env.action_space}")
    action_dim = int(np.prod(env.action_space.shape))
    if action_dim != expected_action_dim:
        raise RuntimeError(f"Action dim mismatch: {action_dim} vs {expected_action_dim}")
    obs_dim = int(np.prod(env.observation_space.shape))
    status  = "OK" if obs_dim == expected_obs_dim else f"WARNING: expected {expected_obs_dim}"
    print(f"[SANITY] obs dim: {obs_dim} ({status})")
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--unity-file",      type=str,  default="HardAngle.exe")
    p.add_argument("--no-graphics",     action="store_true", default=True)
    p.add_argument("--n-envs",          type=int,  default=10)
    p.add_argument("--model-path",      type=str,  default="sac64H.zip")
    p.add_argument("--vecnorm-path",    type=str,  default="sac64H.pkl")
    p.add_argument("--resume",          action="store_true", default=True)
    p.add_argument("--total-timesteps", type=int,  default=14_000_000)
    return p.parse_args()
 
 
def load_vecnormalize(vecnorm_path, env, gamma=0.99, clip_obs=10.0):
    if vecnorm_path and os.path.exists(vecnorm_path):
        try:
            loaded = VecNormalize.load(vecnorm_path, env)
            loaded.training    = True
            loaded.norm_reward = True
            print(f"[MAIN] Načítaný VecNormalize z {vecnorm_path}")
            return loaded
        except Exception as e:
            print(f"[MAIN] VecNormalize načítanie zlyhalo ({e}), vytváram nový.")
    return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=clip_obs, gamma=gamma)
 
 
def main():
    args = parse_args()
    print(f"[MAIN] Pokračovanie z modelu: {args.model_path}")
    print(f"[MAIN] Vytváranie {args.n_envs} prostredí z '{args.unity_file}'")
 
    env_fns = [make_env(args.unity_file, args.no_graphics, worker_id=i) for i in range(args.n_envs)]
    raw_env = SubprocVecEnv(env_fns)
 
    env = load_vecnormalize(args.vecnorm_path, raw_env)
 
    sanity_checks_env(env, expected_action_dim=2, expected_obs_dim=19)

    model = None
    if args.resume and os.path.exists(args.model_path):
        try:
            print(f"[MAIN] Načítavam model z {args.model_path} ...")
            model = SAC.load(
                args.model_path,
                env=env,
                device="auto",
                learning_rate=1e-4,       
                buffer_size=500_000,      
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                ent_coef="auto",          
                target_entropy=-1.0,
                tensorboard_log="./tensorboard_sac_parallel/",
                verbose=1,
            )
            model.learning_starts = 1_000
            print(f"[MAIN] Model načítaný. Replay buffer: {model.replay_buffer.size()} krokov.")
        except Exception as e:
            print(f"[MAIN] Načítanie modelu zlyhalo: {e}")
            print("[MAIN] Spúšťam od nuly.")
            model = None
 
    if model is None:
        print("[MAIN] Vytváram nový SAC model.")
        model = SAC(
            "MlpPolicy", env, verbose=1, device="auto",
            tensorboard_log="./tensorboard_sac_parallel/",
            learning_rate=1e-4,
            buffer_size=500_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            target_entropy=-1.0,
        )
 
    callback = RewardLoggingCallback(verbose=1)
 
    print(f"[MAIN] Tréning začína — {args.total_timesteps} krokov")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=1,
            reset_num_timesteps=False,  
        )
        print("[MAIN] Tréning dokončený.")
        model.save("sac65H.zip")
        try:
            env.save("sac65H.pkl")
        except Exception as e:
            print(f"[MAIN] vecnorm uloženie zlyhalo: {e}")
    except KeyboardInterrupt:
        print("Prerušené. Ukladám...")
        try:
            model.save("sac63H_interrupted.zip")
            env.save("sac63H_interrupted.pkl")
        except Exception as e:
            print(f"[MAIN] Uloženie zlyhalo: {e}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        print("[MAIN] Hotovo.")
 
 
if __name__ == "__main__":
    main()