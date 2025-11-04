import os, numpy as np, pandas as pd

SAVE_DIR = "ppo_lander_cont"
eval_npz = np.load(os.path.join(SAVE_DIR, "eval", "evaluations.npz"))
ts = eval_npz["timesteps"]                     # shape: [E]
eval_means = eval_npz["results"].mean(axis=1)  # shape: [E]
eval_stds  = eval_npz["results"].std(axis=1)

best_idx = int(np.argmax(eval_means))
best_mean, best_std, best_step = float(eval_means[best_idx]), float(eval_stds[best_idx]), int(ts[best_idx])

# time to threshold (first eval ≥ 200)
thr = 200.0
thr_idx = np.argmax(eval_means >= thr) if np.any(eval_means >= thr) else None
time_to_200 = int(ts[thr_idx]) if thr_idx is not None else None

# final window stability (last K evals)
K = min(5, len(eval_means))
final_mean = float(np.mean(eval_means[-K:]))
final_std  = float(np.mean(eval_stds[-K:]))

print(f"Best mean ± std: {best_mean:.1f} ± {best_std:.1f} @ {best_step:,} steps")
print(f"Final mean ± std (last {K} evals): {final_mean:.1f} ± {final_std:.1f}")
print("Time to 200:", f"{time_to_200:,} steps" if time_to_200 is not None else "not reached")