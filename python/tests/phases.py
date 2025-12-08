import numpy as np


def _parse_action(action):
    """
    Robust small helper: returns (throttle_idx, steer_idx, handbrake_idx)
    Works for lists, np.arrays, nested, or already flat arrays.
    """
    try:
        a = np.asarray(action).reshape(-1)
    except Exception:
        a = np.array([action])
    # safe defaults
    throttle_idx = int(a[0]) if a.size > 0 else 1
    steer_idx = int(a[1]) if a.size > 1 else 1
    hb_idx = int(a[2]) if a.size > 2 else 0
    return throttle_idx, steer_idx, hb_idx


def compute_reward_simple_forward(prev_obs, action, obs):
    """
    PHASE 1 REWARD (0-1M steps): Dead simple forward movement learning.
    Goal: Learn that throttle = forward speed.

    This is the ideal starting reward for 3-5M timestep training.
    """
    if obs is None or len(obs) == 0:
        return 0.0

    # Get speed (assuming first index, adjust if needed)
    speed = abs(obs[0]) if len(obs) > 0 else 0.0

    reward = 0.0

    # PRIMARY REWARD: Moving forward
    reward += speed * 0.5

    # Light penalty for standing still (encourages exploration)
    if speed < 0.1:
        reward -= 0.05

    return float(np.clip(reward, -1.0, 5.0))


def compute_reward_intermediate(prev_obs, action, obs):
    """
    PHASE 2 REWARD (1-3M steps): Add basic collision detection and goal awareness.
    Use this after the agent learns basic forward movement.
    """
    if obs is None or len(obs) == 0:
        return 0.0

    speed = abs(obs[0]) if len(obs) > 0 else 0.0
    reward = 0.0

    # Check for collision (sudden stop)
    if prev_obs is not None and len(prev_obs) > 0:
        prev_speed = abs(prev_obs[0])
        if prev_speed > 0.5 and speed < 0.1:
            return -1.0  # Collision penalty

    # Check for goal (if you have goal detection in observations)
    # Adjust indices based on your observation structure
    if len(obs) > 10:  # Example: assuming goal info is in later indices
        for i in range(10, min(18, len(obs))):
            if obs[i] > 0.9:  # Goal reached heuristic
                return 5.0

    # PRIMARY REWARD: Moving forward
    reward += speed * 0.5

    # Progress reward (if distance to goal decreases)
    if prev_obs is not None and len(obs) > 1:
        for i in range(1, min(10, len(obs))):
            if obs[i] > 0 and prev_obs[i] > 0:
                improvement = prev_obs[i] - obs[i]
                if improvement > 0:
                    reward += improvement * 1.5
                    break

    # Small penalty for standing still
    if speed < 0.1:
        reward -= 0.05

    return float(np.clip(reward, -1.0, 5.0))


def compute_reward_advanced(prev_obs, action, obs):
    """
    PHASE 3 REWARD (3-5M steps): Add action shaping and efficiency rewards.
    Use this for final refinement after basic behavior is learned.
    """
    if obs is None:
        return 0.0

    n = len(obs)
    if n < 6:
        return -0.01

    raycount = n - 6

    def at(i, default=0.0):
        return float(obs[i]) if 0 <= i < n else default

    speed = abs(at(raycount))
    norm_dist = at(raycount + 1)
    signed_angle = at(raycount + 2)

    reward = 0.0

    # Parse action
    throttle_idx, steer_idx, hb_idx = _parse_action(action)
    throttle = -1.0 if throttle_idx == 0 else (0.0 if throttle_idx == 1 else 1.0)
    steer = steer_idx - 1.0
    handbrake = hb_idx == 1

    # Collision check
    if prev_obs is not None:
        prev_speed = abs(prev_obs[raycount]) if len(prev_obs) > raycount else 0.0
        if prev_speed > 1.0 and speed < 0.2:
            return -1.0

    # Goal detection
    if 0 <= norm_dist < 0.05:
        return 5.0

    # Movement reward
    reward += speed * 0.4

    # Progress toward goal
    if prev_obs is not None and len(prev_obs) > raycount + 1:
        prev_dist = float(prev_obs[raycount + 1])
        if prev_dist > 0 and prev_dist - norm_dist > 1e-6:
            reward += (prev_dist - norm_dist) * 2.0

    # Facing goal reward
    reward += max(0.0, (1.0 - abs(signed_angle))) * 0.05

    # MILD action shaping (much lighter than original)
    reward -= abs(steer) * 0.01  # Reduced from 0.02

    if throttle > 0.9 and speed < 0.03:
        reward -= 0.02  # Reduced from 0.05

    if handbrake and norm_dist > 0.2:
        reward -= 0.01  # Reduced from 0.03

    # Light time penalty
    reward -= 0.005

    reward = float(np.clip(reward, -5.0, 5.0))
    return reward


def compute_reward_verbose(prev_obs, action, obs):
    """
    Debug version with detailed logging - uses simple forward reward.
    """
    if obs is None or len(obs) == 0:
        print("[Reward VERBOSE] No observation, returning 0")
        return 0.0

    speed = abs(obs[0]) if len(obs) > 0 else 0.0

    # Parse action
    throttle_idx, steer_idx, hb_idx = _parse_action(action)
    throttle = -1.0 if throttle_idx == 0 else (0.0 if throttle_idx == 1 else 1.0)
    steer = steer_idx - 1.0
    handbrake = hb_idx == 1

    print(f"[Reward VERBOSE] Speed: {speed:.3f}")
    print(
        f"[Reward VERBOSE] Action: throttle_idx={throttle_idx} ({throttle:.1f}), steer_idx={steer_idx} ({steer:.1f}), hb={handbrake}"
    )

    reward = 0.0

    # Speed reward
    speed_reward = speed * 0.5
    reward += speed_reward
    print(f"[Reward VERBOSE] Speed reward: +{speed_reward:.3f}")

    # Standing penalty
    if speed < 0.1:
        reward -= 0.05
        print(f"[Reward VERBOSE] Standing still penalty: -0.05")

    print(f"[Reward VERBOSE] Total reward: {reward:.3f}\n")

    return float(np.clip(reward, -1.0, 5.0))


# ============================================================
# DEFAULT EXPORT - CHANGE THIS AS YOU PROGRESS THROUGH PHASES
# ============================================================

# PHASE 1 (0-1M steps): Use this first!
compute_reward = compute_reward_simple_forward

# PHASE 2 (1-3M steps): Uncomment this after Phase 1 succeeds
# compute_reward = compute_reward_intermediate

# PHASE 3 (3-5M steps): Uncomment this for final refinement
# compute_reward = compute_reward_advanced

# DEBUG: Uncomment this if you need detailed logging
# compute_reward = compute_reward_verbose


# Legacy function name for backward compatibility
compute_reward_scaled = compute_reward
