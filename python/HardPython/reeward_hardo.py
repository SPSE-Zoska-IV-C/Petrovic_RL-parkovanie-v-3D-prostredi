# hard
import math
import numpy as np

GAMMA = 0.99


def compute_reward_continuous(prev_obs, action, obs, info):

    GOAL_REWARD     = 3_000.0
    CRASH_PENALTY   =  -200.0   
    TIMEOUT_PENALTY =  -100.0

    LIVING_BASE = -0.01
    LIVING_NEAR = -0.04

    APPROACH_SCALE = 30.0
    RETREAT_SCALE  = 60.0
    PROGRESS_CLIP  =  3.0

    EXP_SCALE = 30.0
    EXP_SIGMA =  0.12   

    MILESTONE_THRESHOLDS = [0.50, 0.25, 0.15, 0.12, 0.10, 0.05]
    MILESTONE_BONUSES    = [10.0, 50.0, 80.0, 120.0, 300.0, 500.0]

    RAY_INDICES       = list(range(12))
    RAY_DANGER_THRESH = 0.20
    RAY_PENALTY_SCALE = -0.05
    RAY_PENALTY_CLIP  = -0.40

    REVERSE_PENALTY   = -0.50
    REVERSE_THRESH    =  0.0
    REVERSE_FREE_DIST =  0.12   

    HEADING_MAX       = 0.08
    HEADING_FREE_DIST = 0.25

    IDX_NORMDIST = -6
    IDX_ANGLE    = -5
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

    raw_obs = safe_array(info.get("_raw_obs", None))
    if raw_obs is None:
        raw_obs = safe_array(obs)
    if raw_obs is None:
        return 0.0

    prev     = safe_array(prev_obs)
    end_flag = safe_get(raw_obs, IDX_FLAG, 0.0)

    crashed = bool(info.get("crash") or info.get("crashed") or info.get("collision"))
    if not crashed and end_flag <= -0.5:
        crashed = True

    goal_reached = bool(info.get("goal") or info.get("goal_reached"))
    if not goal_reached and end_flag >= 0.5:
        goal_reached = True

    truncated       = bool(info.get("TimeLimit.truncated") or info.get("truncated") or info.get("timeout"))
    terminated_here = bool(info.get("terminated") or info.get("done"))

    episode_reward_so_far = float(info.get("_episode_reward_so_far", 0.0))

    if crashed:
        info["termination_reason"] = "crash"
        total = float(CRASH_PENALTY) - episode_reward_so_far
        if debug:
            print(f"[REWARD] crash | abolish={-episode_reward_so_far:.2f} -> net={total:.2f}")
        return total

    if goal_reached:
        info["termination_reason"] = "goal"
        if debug:
            print(f"[REWARD] goal -> {GOAL_REWARD}")
        return float(GOAL_REWARD)

    if truncated or (terminated_here and not goal_reached):
        info["termination_reason"] = "timeout"
        total = float(TIMEOUT_PENALTY) - episode_reward_so_far
        if debug:
            print(f"[REWARD] timeout | abolish={-episode_reward_so_far:.2f} -> net={total:.2f}")
        return total

    cur_dist = float(np.clip(safe_get(raw_obs, IDX_NORMDIST, 1.0), 1e-4, 1.0))
    angle    = safe_get(raw_obs, IDX_ANGLE, 0.0)

    reward = LIVING_BASE + LIVING_NEAR * (1.0 - cur_dist)

    # Asymetrický progress
    if prev is not None and prev.shape[0] == raw_obs.shape[0]:
        prev_dist = float(np.clip(safe_get(prev, IDX_NORMDIST, cur_dist), 1e-4, 1.0))
        delta     = prev_dist - cur_dist
        scale     = APPROACH_SCALE if delta >= 0 else RETREAT_SCALE
        reward   += float(np.clip(scale * delta, -PROGRESS_CLIP, PROGRESS_CLIP))

        # PBRS proximity
        phi_cur   = EXP_SCALE * math.exp(-cur_dist  / EXP_SIGMA)
        phi_prev  = EXP_SCALE * math.exp(-prev_dist / EXP_SIGMA)
        pbrs      = GAMMA * phi_cur - phi_prev
        reward   += float(np.clip(pbrs, -2.0, 5.0))
    else:
        phi_cur  = EXP_SCALE * math.exp(-cur_dist / EXP_SIGMA)
        baseline = EXP_SCALE * math.exp(-1.0 / EXP_SIGMA)
        reward  += float(np.clip(phi_cur - baseline, 0.0, EXP_SCALE))

    # Wall penalty
    wall_p   = 0.0
    n_danger = 0
    for idx in RAY_INDICES:
        ray_val = safe_get(raw_obs, idx, 1.0)
        if ray_val < RAY_DANGER_THRESH:
            wall_p   += RAY_PENALTY_SCALE * (RAY_DANGER_THRESH - ray_val) / RAY_DANGER_THRESH
            n_danger += 1
    reward += float(np.clip(wall_p, RAY_PENALTY_CLIP, 0.0))
    if debug and n_danger > 0:
        print(f"[REWARD] wall={wall_p:.4f} n_danger={n_danger}")

    # Reverse penalty
    if cur_dist > REVERSE_FREE_DIST:
        try:
            if action is not None:
                a_arr = np.asarray(action, dtype=np.float32).reshape(-1)
                if a_arr.size >= 2 and float(a_arr[1]) < REVERSE_THRESH:
                    reward += float(REVERSE_PENALTY)
                    if debug:
                        print(f"[REWARD] reverse={REVERSE_PENALTY} throttle={float(a_arr[1]):.3f}")
        except Exception:
            pass

    # Heading
    h_scale = HEADING_MAX if cur_dist >= HEADING_FREE_DIST else HEADING_MAX * (cur_dist / HEADING_FREE_DIST)
    reward += float(np.clip(h_scale * (1.0 - abs(angle)), 0.0, HEADING_MAX))

    # Milestones 
    milestones_hit: set = info.get("_milestones_hit", set())
    for threshold, bonus in zip(MILESTONE_THRESHOLDS, MILESTONE_BONUSES):
        if threshold not in milestones_hit and cur_dist <= threshold:
            milestones_hit.add(threshold)
            reward += bonus
            if debug:
                print(f"[REWARD] MILESTONE d<={threshold:.2f} -> +{bonus:.1f} | dist={cur_dist:.4f}")
    info["_milestones_hit"] = milestones_hit

    if debug:
        print(f"[REWARD] step={reward:.4f} dist={cur_dist:.4f} angle={angle:.3f} milestones={milestones_hit}")

    return float(reward)
