# test_quick.py
from wrapper import UnityGymnasiumEnv

env = UnityGymnasiumEnv(
    file_name="C:/Users/TheTr/maturita/builds/ActionEatest",
    no_graphics=False,
    worker_id=0,
    timeout=60,
)

print("\nTesting 1 episode...")
obs, info = env.reset()
print(f"Initial obs shape: {obs.shape}")

for step in range(2100):  # More than 2048
    action = [1, 2, 0]  # Forward
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        reason = "TERMINATED (goal/crash)" if terminated else "TRUNCATED (max steps)"
        print(f"✓ Episode ended at step {step+1}: {reason}")
        break

if step >= 2099:
    print("✗ ERROR: Episode never ended!")

env.close()
