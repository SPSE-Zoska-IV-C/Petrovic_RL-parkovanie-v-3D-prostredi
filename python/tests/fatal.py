# inspect_playback.py
import os, glob, time, csv, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Adjust these if needed
MODEL_NAME_PREFIX = "ppo_unity_interrupted"  # will match ppo_interrupted.zip or ppo_interrupted.*
STATS_GLOB = "vec_normalize_stats*.pkl"  # common save name patterns
UNITY_FILE_DEFAULT = "C:/Users/TheTr/maturita/builds/nedavamTo.exe"  # None -> connect to Editor, or provide exe path string
WORKER_ID = 10  # change if port collision

MAX_STEPS = 20000
SAVE_CSV = "playback_debug.csv"
VERBOSE = True


# ---- helpers to find files ----
def find_model_file(prefix=MODEL_NAME_PREFIX):
    # SB3 saves models as "<prefix>.zip"
    cand = glob.glob(f"{prefix}*")
    # prefer .zip
    zipc = [c for c in cand if c.endswith(".zip")]
    if zipc:
        return zipc[0]
    if cand:
        return cand[0]
    return None


def find_stats_file(glob_pattern=STATS_GLOB):
    found = glob.glob(glob_pattern)
    if found:
        return found[0]
    return None


# ---- user: replace make_env import if needed ----
# This script expects a function make_env(unity_file, no_graphics, worker_id) in the same repo.
# If your training script defines make_env in a module, import it:
try:
    from train import make_env  # if your main training file is train.py
except Exception:
    try:
        from __main__ import make_env  # fallback if you run from same file (unlikely)
    except Exception:
        # Fallback: try to import from the file where you had wrapper
        try:
            from your_training_module import make_env
        except Exception:
            print(
                "Could not import make_env automatically. Ensure this script is run from project root and 'make_env' is defined in train.py or modify this script to import correctly."
            )
            raise

# ---- locate files ----
model_file = find_model_file()
stats_file = find_stats_file()

print("Model file found:", model_file)
print("VecNormalize stats file found:", stats_file)

if model_file is None:
    raise SystemExit(
        "No model file found with prefix '{}'. Rename or move model into this folder.".format(
            MODEL_NAME_PREFIX
        )
    )

# ---- create env (visual) ----
unity_file = UNITY_FILE_DEFAULT
no_graphics = False  # you said you want to watch
venv = DummyVecEnv([make_env(unity_file, no_graphics, WORKER_ID)])

# try load normalize stats if present
if stats_file:
    try:
        venv = VecNormalize.load(stats_file, venv)
        venv.training = False
        venv.norm_reward = False
        print("Loaded VecNormalize stats:", stats_file)
    except Exception as e:
        print("Failed to load VecNormalize stats:", e)
        print("Proceeding without stats (observations may be scaled differently).")

# ---- load model ----
print("Loading model:", model_file)
model = PPO.load(model_file, env=venv)

# ---- run one episode and log ----
obs = venv.reset()
# deduce speed index heuristically
obs0 = np.asarray(obs)[0] if isinstance(obs, (list, tuple, np.ndarray)) else np.asarray(obs)
obs_len = obs0.shape[-1] if hasattr(obs0, "shape") else len(obs0)
speed_index = obs_len - 6 if obs_len >= 6 else 0
print(f"Inferred obs length={obs_len}, using speed_index={speed_index}")

rows = []
prev_obs = None
episode_reward = 0.0


def anomaly(o):
    a = np.asarray(o, dtype=np.float64)
    if np.any(np.isnan(a)) or np.any(np.isinf(a)):
        return "nan_or_inf"
    if np.max(np.abs(a)) > 1e6:
        return "huge_vals"
    return ""


start = time.time()
for step in range(MAX_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    # step env
    obs, reward, done, info = venv.step(action)
    # vectorized unwrap
    if isinstance(obs, (list, tuple, np.ndarray)):
        obs0 = np.asarray(obs)[0]
    else:
        obs0 = np.asarray(obs)
    # handle reward and info shapes
    r = (
        float(np.asarray(reward)[0])
        if isinstance(reward, (list, tuple, np.ndarray))
        else float(reward)
    )
    episode_reward += r
    done_flag = False
    info0 = {}
    if isinstance(done, (list, tuple, np.ndarray)):
        done_flag = bool(done[0])
    else:
        done_flag = bool(done)
    if isinstance(info, (list, tuple, np.ndarray)):
        info0 = info[0] if len(info) > 0 else {}
    else:
        info0 = info or {}

    # check anomalies
    an = anomaly(obs0)
    # attempt to read speed safely
    speed_val = float(obs0[speed_index]) if len(obs0) > speed_index else float(obs0[0])
    # the actual action applied to env (clean)
    a_clean = np.asarray(action)
    if a_clean.ndim > 1:
        a_for0 = a_clean[0].tolist()
    else:
        a_for0 = a_clean.tolist()

    rows.append(
        {
            "step": step,
            "action": a_for0,
            "speed": speed_val,
            "reward": r,
            "done": done_flag,
            "anomaly": an,
            "info": info0,
            "obs_sample": obs0.tolist()[:12],  # first 12 values preview
        }
    )

    # print diagnostic for early steps and when things go weird
    if step < 40 or abs(r) > 10 or an or done_flag:
        print(
            f"[S{step:04d}] act={a_for0} speed={speed_val:.5f} r={r:.5f} done={done_flag} an={an}"
        )

    if done_flag or ("episode" in info0 and info0.get("episode") is not None):
        ep = info0.get("episode", None)
        if ep:
            print("[EP-END] info episode:", ep)
        else:
            print(f"[EP-END] done at step {step} episode_reward={episode_reward:.3f}")
        break

    prev_obs = obs0

elapsed = time.time() - start
# save csv
with open(SAVE_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["step", "action", "speed", "reward", "done", "anomaly", "info", "obs_sample"]
    )
    writer.writeheader()
    for r in rows:
        # info field can be non-serializable; coerce to str
        r2 = r.copy()
        r2["info"] = str(r2["info"])
        writer.writerow(r2)

print(
    f"Saved playback trace to {SAVE_CSV} steps={len(rows)} elapsed={elapsed:.2f}s episode_reward={episode_reward:.3f}"
)
venv.close()
