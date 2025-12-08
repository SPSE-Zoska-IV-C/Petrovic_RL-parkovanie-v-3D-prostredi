# # unity_gymnasium_wrapper.py
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# import time
# import traceback
# from mlagents_envs.environment import UnityEnvironment, UnityEnvironmentException
# from mlagents_envs.base_env import ActionTuple

# class UnityGymnasiumEnv(gym.Env):
#     def __init__(self, file_name, no_graphics=True, worker_id=5, timeout=30):
#         print(f"[Wrapper] Attempting to connect to Unity. file_name={file_name}, worker_id={worker_id}, timeout={timeout}s")
#         try:
#             # Use a short timeout for debugging so we fail fast if Unity isn't ready.
#             self.unity_env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics, worker_id=worker_id, timeout_wait=timeout)
#         except Exception as e:
#             print("[Wrapper] Failed to create UnityEnvironment. Exception:")
#             traceback.print_exc()
#             raise

#         print("[Wrapper] UnityEnvironment() returned without exception. Calling reset()...")
#         try:
#             self.unity_env.reset()
#         except Exception as e:
#             print("[Wrapper] unity_env.reset() failed:")
#             traceback.print_exc()
#             raise

#         print("[Wrapper] Connected and scene reset successfully.")

#         # Automatically pick the first available behavior
#         available_behaviors = list(self.unity_env.behavior_specs.keys())
#         print(f"[Wrapper] Available behaviors: {available_behaviors}")

#         if len(available_behaviors) == 0:
#             raise RuntimeError("No behaviors found in Unity environment! Make sure your agent has a Behavior Parameters component.")

#         self.behavior_name = available_behaviors[0]
#         print(f"[Wrapper] Using behavior: '{self.behavior_name}'")
#         spec = self.unity_env.behavior_specs[self.behavior_name]

#         # build action_space
#         a_spec = spec.action_spec
#         # continuous actions?
#         if a_spec.continuous_size > 0 and a_spec.discrete_size == 0:
#             self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(a_spec.continuous_size,), dtype=np.float32)
#             self._is_continuous = True
#         else:
#             # Use MultiDiscrete to match Unity's discrete branches exactly.
#             # a_spec.discrete_branches is e.g. [3,3,2] (steer, throttle, handbrake)
#             self._is_continuous = False
#             # store branch sizes as numpy array for convenience
#             self._disc_branch_sizes = np.array(a_spec.discrete_branches, dtype=np.int64)
#             self.action_space = spaces.MultiDiscrete(self._disc_branch_sizes)

#         # observation space: concatenate all vector observations into 1D array
#         # Try both old and new API
#         if hasattr(spec, 'observation_shapes'):
#             obs_shapes = spec.observation_shapes
#         elif hasattr(spec, 'observation_specs'):
#             obs_shapes = [obs_spec.shape for obs_spec in spec.observation_specs]
#         else:
#             raise RuntimeError("Cannot determine observation shapes from BehaviorSpec")

#         total_obs = sum([int(np.prod(s)) for s in obs_shapes])
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs,), dtype=np.float32)

#         # keep last obs for reward fn convenience
#         self._last_obs = None

#         # quick debug so you can verify shapes on connect
#         print(f"[UnityGymnasiumEnv] behavior: {self.behavior_name}")
#         print(f"[UnityGymnasiumEnv] observation shapes: {obs_shapes} -> total {total_obs}")
#         print(f"[UnityGymnasiumEnv] action_spec: continuous_size={a_spec.continuous_size}, discrete_branches={a_spec.discrete_branches}")
#         print(f"[UnityGymnasiumEnv] action_space: {self.action_space}")

#     def _concat_obs(self, obs_list):
#         parts = [np.asarray(o).ravel() for o in obs_list]
#         if len(parts) == 0:
#             return np.zeros(self.observation_space.shape, dtype=np.float32)
#         return np.concatenate(parts).astype(np.float32)

#     def reset(self, seed=None, options=None):
#         self.unity_env.reset()
#         decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
#         if len(decision_steps) == 0:
#             # no agent present — return zeros
#             obs = np.zeros(self.observation_space.shape, dtype=np.float32)
#             self._last_obs = obs
#             return obs, {}
#         # single agent: take first
#         agent_id = list(decision_steps.agent_id)[0]
#         dec = decision_steps[agent_id]
#         obs = self._concat_obs(dec.obs)
#         self._last_obs = obs
#         return obs, {}

#     def step(self, action):
#         if self._is_continuous:
#             continuous = np.array(action, dtype=np.float32).reshape(1, -1)
#             discrete = np.empty((1, 0), dtype=np.int32)
#         else:
#             action_vec = np.asarray(action, dtype=np.int32).reshape(-1)
#             expected = int(self._disc_branch_sizes.size)
#             if action_vec.size != expected:
#                 raise ValueError(f"Expected action with {expected} discrete branches, got {action_vec.size}. Action: {action}")
#             continuous = np.empty((1, 0), dtype=np.float32)
#             discrete = action_vec.reshape(1, -1)

#         action_tuple = ActionTuple(continuous=continuous, discrete=discrete)
#         self.unity_env.set_actions(self.behavior_name, action_tuple)
#         self.unity_env.step()

#         decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

#         # Prepare info dict
#         info = {}
#         prev_obs = self._last_obs

#         if len(terminal_steps) > 0:
#             agent_id = list(terminal_steps.agent_id)[0]
#             term = terminal_steps[agent_id]
#             obs = self._concat_obs(term.obs)

#             # detect truncated vs terminated
#             interrupted = term.interrupted if hasattr(term, 'interrupted') else False
#             terminated = not interrupted
#             truncated = interrupted

#             if truncated:
#                 info["TimeLimit.truncated"] = True

#             self._last_obs = obs
#             # Return previous obs in info so reward function can use it
#             info["prev_obs"] = prev_obs
#             return obs, 0.0, terminated, truncated, info

#         elif len(decision_steps) > 0:
#             agent_id = list(decision_steps.agent_id)[0]
#             dec = decision_steps[agent_id]
#             obs = self._concat_obs(dec.obs)
#             self._last_obs = obs
#             info["prev_obs"] = prev_obs
#             return obs, 0.0, False, False, info

#         else:
#             # fallback: no agents (weird)
#             obs = np.zeros(self.observation_space.shape, dtype=np.float32)
#             self._last_obs = obs
#             info["prev_obs"] = prev_obs
#             return obs, 0.0, True, False, info

#     def close(self):
#         self.unity_env.close()

# unity_gymnasium_wrapper.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traceback
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


class UnityGymnasiumEnv(gym.Env):
    """
    Gymnasium wrapper around ML-Agents UnityEnvironment.
    - Supports continuous or discrete action specs.
    - For continuous actions and continuous_size == 3, maps:
        [steer, throttle, handbrake_raw] -> handbrake_mapped = (handbrake_raw + 1) / 2
      so the agent learns handbrake in 0..1 range (CarAgent clamps to 0..1 anyway).
    - Stores and returns prev_obs and prev_action in info for reward functions.
    """

    def __init__(self, file_name=None, no_graphics=True, worker_id=5, timeout=30, debug=False):
        self.debug = debug
        print(
            f"[Wrapper] Connecting to Unity: file_name={file_name}, worker_id={worker_id}, timeout={timeout}s"
        )
        try:
            self.unity_env = UnityEnvironment(
                file_name=file_name,
                no_graphics=no_graphics,
                worker_id=worker_id,
                timeout_wait=timeout,
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
            raise RuntimeError(
                "No behaviors found in Unity environment! Make sure BehaviorParameters is present."
            )

        self.behavior_name = available_behaviors[0]
        spec = self.unity_env.behavior_specs[self.behavior_name]

        # action_space construction
        a_spec = spec.action_spec
        self._is_continuous = a_spec.continuous_size > 0 and a_spec.discrete_size == 0

        if self._is_continuous:
            cont_size = int(a_spec.continuous_size)
            # If cont_size == 3 (steer, throttle, handbrake), set per-dim bounds so handbrake is [0,1]
            if cont_size == 3:
                low = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
                high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
            else:
                # generic fallback: all dims in [-1,1]
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(cont_size,), dtype=np.float32
                )
        else:
            # discrete case
            self._disc_branch_sizes = np.array(a_spec.discrete_branches, dtype=np.int64)
            self.action_space = spaces.MultiDiscrete(self._disc_branch_sizes)

        # observation space: concatenate all vector observations into 1D array
        if hasattr(spec, "observation_shapes"):
            obs_shapes = spec.observation_shapes
        elif hasattr(spec, "observation_specs"):
            obs_shapes = [o.shape for o in spec.observation_specs]
        else:
            raise RuntimeError("Cannot determine observation shapes from BehaviorSpec")

        total_obs = sum([int(np.prod(s)) for s in obs_shapes])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs,), dtype=np.float32
        )

        # internal caches
        self._last_obs = None
        self._last_action = None  # 1D numpy array (raw action as sent to Unity)
        self._step_debug_count = 0

        # debug print
        print(f"[UnityGymnasiumEnv] behavior: {self.behavior_name}")
        print(f"[UnityGymnasiumEnv] obs_shapes: {obs_shapes} -> total_obs={total_obs}")
        print(
            f"[UnityGymnasiumEnv] action_spec: continuous_size={a_spec.continuous_size}, discrete_branches={a_spec.discrete_branches}"
        )
        print(f"[UnityGymnasiumEnv] action_space: {self.action_space}")

    def _concat_obs(self, obs_list):
        parts = [np.asarray(o).ravel() for o in obs_list]
        if len(parts) == 0:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def reset(self, seed=None, options=None):
        # reset Unity and fetch initial decision step
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
        """
        action: either a 1D array/list of floats (continuous) or discrete branches for MultiDiscrete.
        Returns: obs, reward(=0.0), terminated, truncated, info
        Note: reward is computed in Python wrapper or training loop. This wrapper returns prev_obs and prev_action in info.
        """
        # preserve previous caches for info
        prev_obs = self._last_obs
        prev_action = None if self._last_action is None else self._last_action.copy()

        # format action into ActionTuple
        if self._is_continuous:
            continuous = np.asarray(action, dtype=np.float32).reshape(1, -1)
            # If actor outputs third dim in [0,1] already, this mapping keeps it; if actor outputs -1..1 we map it below.
            # Ensure lengths match expected
            if continuous.shape[1] != self.action_space.shape[0]:
                # allow cases where action_space shape is vector instead of per-dim low/high shape:
                # if action_space.shape[0] differs only because we used per-dim low/high for 3-dim case, still accept cont_size from unity
                # but we will sanity-check by comparing to BehaviorSpec if needed
                pass

            # Normalize/map the 3rd axis (handbrake) to [0,1] if cont dim is 3
            if continuous.shape[1] >= 3:
                # continuous[:,2] expected from policy in box [-1,1] — map to [0,1] for Unity (CarAgent clamps to 0..1)
                # But if action_space already declared third dim low=0, high=1, then incoming values should be in 0..1.
                # We'll defensively handle both cases: if values are <=1 and >=0 we keep them; otherwise map from [-1,1] to [0,1].
                hb_raw = continuous[:, 2]
                # detect if likely in [-1,1] centered regime (mean near 0), map; else assume already 0..1
                if np.any(hb_raw < 0.0) or np.any(hb_raw > 1.0):
                    hb_mapped = np.clip((hb_raw + 1.0) / 2.0, 0.0, 1.0)
                    continuous[:, 2] = hb_mapped
                else:
                    # ensure small numerical safety
                    continuous[:, 2] = np.clip(hb_raw, 0.0, 1.0)

            discrete = np.empty((1, 0), dtype=np.int32)
            # store last_action as 1D python array (floats) - the *mapped* action that goes to Unity
            self._last_action = continuous.ravel().astype(np.float32)
        else:
            action_vec = np.asarray(action, dtype=np.int32).reshape(-1)
            expected = int(self._disc_branch_sizes.size)
            if action_vec.size != expected:
                raise ValueError(
                    f"Expected action with {expected} discrete branches, got {action_vec.size}. Action: {action}"
                )
            continuous = np.empty((1, 0), dtype=np.float32)
            discrete = action_vec.reshape(1, -1)
            self._last_action = discrete.ravel().astype(np.int32)

        # debug: show the first few actions we send
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
        # prev_action as plain list (JSON-friendly) or None
        info["prev_action"] = None if prev_action is None else prev_action.tolist()

        # termination/truncation handling
        if len(terminal_steps) > 0:
            agent_id = list(terminal_steps.agent_id)[0]
            term = terminal_steps[agent_id]
            obs = self._concat_obs(term.obs)

            interrupted = term.interrupted if hasattr(term, "interrupted") else False
            terminated = not interrupted
            truncated = interrupted

            if truncated:
                info["TimeLimit.truncated"] = True

            self._last_obs = obs
            return obs, 0.0, terminated, truncated, info

        if len(decision_steps) > 0:
            agent_id = list(decision_steps.agent_id)[0]
            dec = decision_steps[agent_id]
            obs = self._concat_obs(dec.obs)
            self._last_obs = obs
            return obs, 0.0, False, False, info

        # fallback
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._last_obs = obs
        return obs, 0.0, True, False, info

    def close(self):
        self.unity_env.close()
