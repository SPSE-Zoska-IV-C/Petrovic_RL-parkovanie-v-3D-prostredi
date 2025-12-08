# import numpy as np

# def _parse_action_continuous(action):
#     """
#     Interpret continuous action in Unity order: [steer, throttle, handbrake_float]
#     steer and throttle are already in [-1, +1].
#     handbrake_float is in [0,1] (or maybe [-1,1] depending on your design).
#     Returns continuous values directly.
#     """
#     a = np.asarray(action, dtype=np.float32).reshape(-1)
#     # Ensure the length is 3
#     if a.size < 3:
#         raise ValueError(f"Expected action of shape (3,), got {a}")

#     steer = float(a[0])
#     throttle = float(a[1])
#     handbrake_float = float(a[2])
#     # If your Unity clamps hb to 0..1, you might want to threshold or use it directly:
#     handbrake = handbrake_float  # keep as continuous

#     return steer, throttle, handbrake_float, handbrake

# def compute_reward_continuous(prev_obs, action, obs):
#     """
#     Reward function adapted to continuous action space.
#     """
#     if obs is None:
#         return 0.0

#     n = len(obs)
#     if n < 6:
#         # fallback / meaningless obs
#         return -0.01

#     # same observation parsing
#     raycount = n - 6
#     def at(i, default=0.0):
#         return float(obs[i]) if 0 <= i < n else default

#     speed = abs(at(raycount))
#     norm_dist = at(raycount + 1)
#     signed_angle = at(raycount + 2)

#     steer, throttle, handbrake_float, handbrake = _parse_action_continuous(action)

#     # 1) Crash detection
#     if prev_obs is not None and len(prev_obs) > raycount:
#         prev_speed = abs(prev_obs[raycount])
#         if prev_speed > 1.0 and speed < 0.2:
#             return float(np.clip(-10.0, -100.0, 100.0))

#     # 2) Goal reached
#     if 0 <= norm_dist < 0.05:
#         return float(np.clip(50.0, -100.0, 100.0))

#     # 3) Progress-based shaping
#     reward = 0.0
#     if prev_obs is not None and len(prev_obs) > raycount + 1:
#         prev_dist = float(prev_obs[raycount + 1])
#         progress = prev_dist - norm_dist
#         if progress > 1e-4:
#             reward += progress * 5.0

#     # 4) Movement reward
#     reward += speed * 0.2

#     # 5) Facing-goal shaping
#     reward += max(0.0, (1.0 - abs(signed_angle))) * 0.02

#     # 6) Steering penalty (continuous)
#     reward -= abs(steer) * 0.02

#     # 7) Wheelspin penalty (throttle continuous)
#     # If throttle is very high and speed very low -> spinning
#     if throttle > 0.9 and speed < 0.03:
#         reward -= 0.5

#     # 8) Handbrake penalty (continuous)
#     # If handbrake_float is high and away from goal, penalize proportionally
#     if handbrake_float > 0.5 and norm_dist > 0.2:
#         # scale penalty by how hard the handbrake is pressed
#         reward -= 0.05 * handbrake_float

#     # 9) Idle / time penalty
#     reward -= 0.01
#     if speed < 0.03:
#         reward -= 0.2 * (1.0 - throttle)  # more throttle when idle should be worse?

#     # Clip
#     return float(np.clip(reward, -100.0, 100.0))

# # You can also make a "simple" version similarly by changing only how you parse action.


# reward_calc.py
import numpy as np


def _safe_asarray_flat(x, dtype=np.float32):
    if x is None:
        return None
    a = np.asarray(x, dtype=dtype)
    if a.ndim > 1:
        a = a.reshape(-1)
    return a


def compute_reward_continuous(prev_obs, action, obs, info=None):
    """
    Continuous-friendly reward.
    prev_obs: previous observation (list/np array) or None
    action: action that was SENT to Unity (python list of floats; note: wrapper sends mapped handbrake already)
    obs: current observation (list/np array)
    info: dict from wrapper; expects info.get('prev_action') possibly set, and TimeLimit.truncated flag
    """
    # basic checks
    if obs is None:
        return 0.0
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n = obs.size
    if n < 6:
        return -0.01

    raycount = n - 6

    def at(i, default=0.0):
        return float(obs[i]) if 0 <= i < n else default

    speed = abs(at(raycount))
    norm_dist = at(raycount + 1)
    signed_angle = at(raycount + 2)

    # parse action (action might already be a list of floats)
    try:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
    except Exception:
        a = np.zeros(3, dtype=np.float32)
    if a.size < 3:
        a = np.pad(a, (0, 3 - a.size), "constant")

    steer = float(np.clip(a[0], -1.0, 1.0))
    throttle = float(np.clip(a[1], -1.0, 1.0))
    # wrapper maps the third action to [0,1] before sending, so handbrake_float is already in [0,1]
    handbrake_float = float(np.clip(a[2], 0.0, 1.0))

    # crash detection
    if prev_obs is not None:
        prev_obs_a = np.asarray(prev_obs, dtype=np.float32).reshape(-1)
        if prev_obs_a.size > raycount:
            prev_speed = abs(prev_obs_a[raycount])
            if prev_speed > 1.0 and speed < 0.2:
                return float(np.clip(-10.0, -100.0, 100.0))

    # goal reached
    if 0 <= norm_dist < 0.05:
        return float(np.clip(50.0, -100.0, 100.0))

    reward = 0.0

    # progress-based shaping
    if prev_obs is not None:
        prev_obs_a = np.asarray(prev_obs, dtype=np.float32).reshape(-1)
        if prev_obs_a.size > raycount + 1:
            prev_dist = float(prev_obs_a[raycount + 1])
            progress = prev_dist - norm_dist
            if progress > 1e-4:
                reward += progress * 5.0

    # movement reward
    reward += speed * 0.2

    # facing-the-goal small shaping
    reward += max(0.0, (1.0 - abs(signed_angle))) * 0.02

    # steering magnitude penalty
    reward -= abs(steer) * 0.02

    # wheelspin penalty
    if throttle > 0.9 and speed < 0.03:
        reward -= 0.5

    # handbrake penalty (proportional)
    if handbrake_float > 0.5 and norm_dist > 0.2:
        reward -= 0.05 * handbrake_float

    # small time penalty + idle penalty
    reward -= 0.01
    if speed < 0.03:
        reward -= 0.2

    # smoothness penalty using prev_action (info['prev_action'] if present)
    if info is not None and "prev_action" in info and info["prev_action"] is not None:
        prev_act = _safe_asarray_flat(info["prev_action"])
        if prev_act is not None and prev_act.size >= 2:
            prev_steer = float(prev_act[0])
            prev_throttle = float(prev_act[1])
            jerk = abs(steer - prev_steer) + abs(throttle - prev_throttle)
            reward -= 0.05 * jerk

    # truncation penalty if episode timed out
    if info is not None and info.get("TimeLimit.truncated", False):
        reward -= 0.5

    return float(np.clip(reward, -100.0, 100.0))


# backward compatibility alias if something imports compute_reward_scaled
compute_reward_scaled = compute_reward_continuous
