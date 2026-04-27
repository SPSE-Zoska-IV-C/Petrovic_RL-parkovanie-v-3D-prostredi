# wrapper script
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traceback
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


class UnityGymnasiumEnv(gym.Env):

    def __init__(self, file_name=None, no_graphics=True, worker_id=5, timeout=30, debug=False):
        self.debug = debug
        print(f"[Wrapper] Connecting to Unity: file_name={file_name}, worker_id={worker_id}, timeout={timeout}s")
        try:
            self.unity_env = UnityEnvironment(
                file_name=file_name,
                no_graphics=no_graphics,
                worker_id=worker_id,
                timeout_wait=timeout
            )
        except Exception:
            print("[Wrapper] UnityEnvironment() creation failed:")
            traceback.print_exc()
            raise

        try:
            self.unity_env.reset()
        except Exception:
            print("[Wrapper] unity_env.reset() failed:")
            traceback.print_exc()
            raise

        available_behaviors = list(self.unity_env.behavior_specs.keys())
        print(f"[Wrapper] Available behaviors: {available_behaviors}")
        if len(available_behaviors) == 0:
            raise RuntimeError("No behaviors found in Unity environment! Make sure BehaviorParameters is present.")

        self.behavior_name = available_behaviors[0]
        spec = self.unity_env.behavior_specs[self.behavior_name]
        
        a_spec = spec.action_spec
        self._is_continuous = (a_spec.continuous_size > 0 and a_spec.discrete_size == 0)

        if self._is_continuous:
            cont_size = int(a_spec.continuous_size)
            if cont_size == 2:
                low = np.array([-1.0, -1.0], dtype=np.float32)
                high = np.array([1.0, 1.0], dtype=np.float32)
                self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
            else:

                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(cont_size,), dtype=np.float32)
        else:
            # discrete case
            self._disc_branch_sizes = np.array(a_spec.discrete_branches, dtype=np.int64)
            self.action_space = spaces.MultiDiscrete(self._disc_branch_sizes)

        if hasattr(spec, 'observation_shapes'):
            obs_shapes = spec.observation_shapes
        elif hasattr(spec, 'observation_specs'):
            obs_shapes = [o.shape for o in spec.observation_specs]
        else:
            raise RuntimeError("Cannot determine observation shapes from BehaviorSpec")

        total_obs = sum([int(np.prod(s)) for s in obs_shapes])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs,), dtype=np.float32)

      
        self._last_obs = None
        self._last_action = None  
        self._step_debug_count = 0

        print(f"[UnityGymnasiumEnv] behavior: {self.behavior_name}")
        print(f"[UnityGymnasiumEnv] obs_shapes: {obs_shapes} -> total_obs={total_obs}")
        print(f"[UnityGymnasiumEnv] action_spec: continuous_size={a_spec.continuous_size}, discrete_branches={a_spec.discrete_branches}")
        print(f"[UnityGymnasiumEnv] action_space: {self.action_space}")

    def _concat_obs(self, obs_list):
        parts = [np.asarray(o).ravel() for o in obs_list]
        if len(parts) == 0:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def reset(self, seed=None, options=None):
        
        self.unity_env.reset()
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        if len(decision_steps) == 0:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            self._last_obs = obs
            return obs, {}
        agent_id = list(decision_steps.agent_id)[0]
        dec = decision_steps[agent_id]
        obs = self._concat_obs(dec.obs)
        self._last_obs = obs
        return obs, {}

    def step(self, action):
        prev_obs = self._last_obs
        prev_action = None if self._last_action is None else self._last_action.copy()

        if self._is_continuous:
            continuous = np.asarray(action, dtype=np.float32).reshape(1, -1)

            discrete = np.empty((1, 0), dtype=np.int32)
            self._last_action = continuous.ravel().astype(np.float32)
        else:
            action_vec = np.asarray(action, dtype=np.int32).reshape(-1)
            expected = int(self._disc_branch_sizes.size)
            if action_vec.size != expected:
                raise ValueError(f"Expected action with {expected} discrete branches, got {action_vec.size}. Action: {action}")
            continuous = np.empty((1, 0), dtype=np.float32)
            discrete = action_vec.reshape(1, -1)
            self._last_action = discrete.ravel().astype(np.int32)

        if self.debug and self._step_debug_count < 20:
            if self._is_continuous:
                print(f"[Wrapper DEBUG] sending continuous -> {self._last_action}")
            else:
                print(f"[Wrapper DEBUG] sending discrete -> {self._last_action}")
            self._step_debug_count += 1

        action_tuple = ActionTuple(continuous=continuous, discrete=discrete)
        self.unity_env.set_actions(self.behavior_name, action_tuple)
        self.unity_env.step()

        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

        info = {}
        info["prev_obs"] = None if prev_obs is None else prev_obs.tolist()
        info["prev_action"] = None if prev_action is None else prev_action.tolist()
        
        if len(terminal_steps) > 0:
            agent_id = list(terminal_steps.agent_id)[0]
            term = terminal_steps[agent_id]
            obs = self._concat_obs(term.obs)

            interrupted = term.interrupted if hasattr(term, 'interrupted') else False
            terminated = not interrupted
            truncated = interrupted

            if truncated:
                info["TimeLimit.truncated"] = True

            self._last_obs = obs

            self._inject_crash_goal_info(obs, info)

            info['terminated'] = bool(terminated)
            info['truncated'] = bool(truncated)

            return obs, 0.0, terminated, truncated, info

        if len(decision_steps) > 0:
            agent_id = list(decision_steps.agent_id)[0]
            dec = decision_steps[agent_id]
            obs = self._concat_obs(dec.obs)
            self._last_obs = obs

            self._inject_crash_goal_info(obs, info)

            info['terminated'] = False
            info['truncated'] = False

            return obs, 0.0, False, False, info

        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._last_obs = obs
        # attempt detection in fallback too (harmless)
        self._inject_crash_goal_info(obs, info)
        info['terminated'] = True
        info['truncated'] = False
        return obs, 0.0, True, False, info

    def _inject_crash_goal_info(self, obs_array, info):
        try:
            _obs_arr = np.asarray(obs_array, dtype=np.float32).reshape(-1)
            total_len = _obs_arr.shape[0]

            end_flag_idx = total_len - 1 if total_len >= 1 else None
            alt_crash_idx = total_len - 2 if total_len >= 2 else None

            info.setdefault('crash', info.get('crash', False))
            info.setdefault('obs_crash_val', info.get('obs_crash_val', 0.0))
            info.setdefault('goal', info.get('goal', False))
            info.setdefault('obs_goal_val', info.get('obs_goal_val', 0.0))

            if end_flag_idx is not None:
                end_val = float(_obs_arr[end_flag_idx])
                if end_val <= -0.5:
                    info['crash'] = True
                    info['obs_crash_val'] = end_val
                    return
                if end_val >= 0.5:
                    info['goal'] = True
                    info['obs_goal_val'] = end_val
                    return

            if alt_crash_idx is not None:
                maybe_val = float(_obs_arr[alt_crash_idx])
                if maybe_val >= 0.5:
                    info['crash'] = True
                    info['obs_crash_val'] = maybe_val
                else:
                    info.setdefault('crash', False)
                    info.setdefault('obs_crash_val', maybe_val)

                if end_flag_idx is not None:
                    last_val = float(_obs_arr[end_flag_idx])
                    if last_val >= 0.5:
                        info['goal'] = True
                        info['obs_goal_val'] = last_val
                    else:
                        info.setdefault('goal', False)
                        info.setdefault('obs_goal_val', last_val)
        except Exception:
            info.setdefault('crash', info.get('crash', False))
            info.setdefault('obs_crash_val', info.get('obs_crash_val', 0.0))
            info.setdefault('goal', info.get('goal', False))
            info.setdefault('obs_goal_val', info.get('obs_goal_val', 0.0))

    def close(self):
        try:
            self.unity_env.close()
        except Exception:
            pass
