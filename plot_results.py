import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

SAVE_DIR = "ppo_lander_cont"

train_df = pd.read_csv(os.path.join(SAVE_DIR, "train_monitor/monitor.csv"), skiprows=1)
plt.plot(train_df["l"].cumsum(), train_df["r"], label="Training curve", alpha=0.6)

data = np.load(os.path.join(SAVE_DIR, "eval/evaluations.npz"))
plt.plot(data["timesteps"], data["results"].mean(axis=1), label="Evaluation curve", linewidth=2)

plt.xlabel("Episode number")
plt.ylabel("Episode reward")
plt.title("PPO LunarLander-v3 (Continuous)")
plt.legend()
plt.grid()
plt.show()