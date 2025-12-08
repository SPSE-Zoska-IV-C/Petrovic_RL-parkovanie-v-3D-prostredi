# try_connect.py
from wrapper import UnityGymnasiumEnv
import traceback

for wid in range(5):
    try:
        print(f"Trying worker_id={wid} ...")
        env = UnityGymnasiumEnv(file_name=None, no_graphics=True, worker_id=wid, timeout=10)
        print(f"Connected with worker_id={wid}")
        env.close()
        break
    except Exception as e:
        print(f"worker_id={wid} failed:")
        traceback.print_exc()
