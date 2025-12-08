# debug_env.py
import numpy as np
from wrapper import UnityGymnasiumEnv  # same wrapper you use
from reward_calc import compute_reward_verbose  # use verbose version

env = UnityGymnasiumEnv(
    file_name="C:/Users/TheTr/maturita/builds/nedavamTo.exe",
    no_graphics=True,
    worker_id=42,
    timeout=300,
)

obs, info = env.reset()
print("OBS SHAPE", np.shape(obs))
print("Obs sample:", obs[:20])

for step in range(200):
    # sample action from action_space to test mapping
    a = env.action_space.sample()
    next_obs, unity_reward, terminated, truncated, info = env.step(a)

    my_reward = compute_reward_verbose(obs, a, next_obs)
    speed = next_obs[0] if len(next_obs) > 0 else None

    print(
        f"step {step:03d} action={a} speed={speed:.4f} unity_reward={unity_reward:.4f} my_reward={my_reward:.4f} terminated={terminated} truncated={truncated}"
    )
    obs = next_obs
    if terminated or truncated:
        print("Episode ended by env at step", step)
        break

env.close()
