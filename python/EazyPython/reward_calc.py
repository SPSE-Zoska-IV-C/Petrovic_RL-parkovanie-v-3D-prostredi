# eazy reward
import numpy as np
import math
 
def compute_reward_continuous(prev_obs, action, obs, info):

    GOAL_REWARD      = 1000.0
    CRASH_PENALTY    = -200.0
    TIMEOUT_PENALTY  = -200.0
 
    LIVING_REWARD    = -0.01
 
    PROGRESS_SCALE   = 20.0
    PROGRESS_CLIP    = 2.0
 
    CLOSE_SCALE      = 5.0
    CLOSE_CLIP       = 5.0

    RAY_INDICES       = list(range(12))
    RAY_DANGER_THRESH = 0.12   
    RAY_PENALTY_SCALE = -0.03   
    RAY_PENALTY_CLIP  = -0.15   
 
    REVERSE_PENALTY   = -0.10
    REVERSE_THRESH    = -0.15
    REVERSE_FREE_DIST = 0.25
 
    IDX_SPEED    = -7   
    IDX_NORMDIST = -6   
    IDX_FLAG     = -1   
 
    debug = bool(info.get("debug_reward", False))
 
    def safe_array(x):
        if x is None:
            return None
        try:
            a = np.asarray(x, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        return np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
 
    def safe_get(arr, idx, default=0.0):
        if arr is None:
            return float(default)
        try:
            v = float(arr[idx])
            if math.isfinite(v):
                return v
        except Exception:
            pass
        return float(default)
 
    raw_obs = safe_array(obs)
 
    cur = safe_array(
        info.get("terminal_observation", None)
        if info.get("terminal_observation", None) is not None
        else obs
    )
    if cur is None:
        return 0.0
 
    prev = safe_array(prev_obs)
 
    cur_dist = safe_get(cur, IDX_NORMDIST, 1.0)
    speed    = safe_get(cur, IDX_SPEED, 0.0)
 
    end_flag = safe_get(raw_obs, IDX_FLAG, 0.0) if raw_obs is not None else 0.0
 
    crashed = bool(info.get("crash") or info.get("crashed") or info.get("collision"))
    if not crashed and end_flag <= -0.5:
        crashed = True
 
    truncated = bool(
        info.get("TimeLimit.truncated") or
        info.get("truncated") or
        info.get("timeout")
    )
    terminated_here = bool(
        info.get("terminated") or
        info.get("done") or
        (info.get("terminal_observation") is not None)
    )
 
    # --- Terminal conditions ---
 
    if crashed:
        if not info.get("crash_handled", False):
            info["crash_handled"] = True
            info["termination_reason"] = "crash"
            if debug:
                print(f"[REWARD DEBUG] crash -> {CRASH_PENALTY}")
            return float(CRASH_PENALTY)
        return 0.0
 
    if info.get("goal_reached", False) or (end_flag >= 0.5):
        info["goal_reached"] = True
        info["termination_reason"] = "goal"
        if debug:
            print(f"[REWARD DEBUG] goal -> {GOAL_REWARD}")
        return float(GOAL_REWARD)
 
    if truncated or (terminated_here and not info.get("goal_reached", False)):
        info["termination_reason"] = "timeout"
        if debug:
            print(f"[REWARD DEBUG] timeout -> {TIMEOUT_PENALTY}")
        return float(TIMEOUT_PENALTY)
 
    # --- Step reward ---
 
    reward = float(LIVING_REWARD)
    progress = 0.0
 
    if prev is not None and prev.shape[0] == cur.shape[0]:
        prev_dist = safe_get(prev, IDX_NORMDIST, None)
        if prev_dist is not None:
            progress = prev_dist - cur_dist
            reward += float(np.clip(PROGRESS_SCALE * progress, -PROGRESS_CLIP, PROGRESS_CLIP))
 
    d = float(np.clip(cur_dist, 0.0, 1.0))
    reward += float(np.clip(-(CLOSE_SCALE * (d ** 2)), -CLOSE_CLIP, 0.0))

    if raw_obs is not None and raw_obs.shape[0] >= 12:
        wall_penalty = 0.0
        n_danger = 0
        for idx in RAY_INDICES:
            ray_val = safe_get(raw_obs, idx, 1.0)
            if ray_val < RAY_DANGER_THRESH:
                closeness = (RAY_DANGER_THRESH - ray_val) / RAY_DANGER_THRESH
                wall_penalty += RAY_PENALTY_SCALE * closeness
                n_danger += 1
        reward += float(np.clip(wall_penalty, RAY_PENALTY_CLIP, 0.0))
        if debug and n_danger > 0:
            print(f"[REWARD DEBUG] wall_penalty={wall_penalty:.4f} n_danger_rays={n_danger}")
 
    if d > REVERSE_FREE_DIST:
        try:
            if action is not None:
                a_arr = np.asarray(action).reshape(-1)
                if a_arr.size >= 2 and float(a_arr[1]) < REVERSE_THRESH:
                    reward += REVERSE_PENALTY
                    if debug:
                        print(f"[REWARD DEBUG] reverse_penalty={REVERSE_PENALTY} dist={d:.3f}")
        except Exception:
            pass
 
    if debug:
        print(
            f"[REWARD DEBUG] reward={reward:.4f} "
            f"progress={progress:.6f} cur_dist={cur_dist:.4f} "
            f"speed={speed:.4f} end_flag={end_flag:.2f}"
        )
 
    return float(reward)