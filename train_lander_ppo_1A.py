import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

ENV_ID = "LunarLander-v3"  
SEED = 20 #20 cuz i just turned 20 :)
SAVE_DIR = "ppo_lander_cont"
N_ENVS = 8

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "train_monitor"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "eval_monitor"), exist_ok=True)

def make_env():
    return gym.make(ENV_ID, continuous=True)

train_env = make_vec_env(make_env, n_envs=N_ENVS, seed=SEED)
train_env = VecMonitor(train_env, os.path.join(SAVE_DIR, "train_monitor"))  

eval_env = Monitor(gym.make(ENV_ID, continuous=True),
                   filename=os.path.join(SAVE_DIR, "eval_monitor", "monitor.csv"))

#using default SB3 hyperparameters 
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    seed=SEED,
    verbose=2,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(SAVE_DIR, "best"),
    log_path=os.path.join(SAVE_DIR, "eval"),
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
)

ckpt_cb = CheckpointCallback(
    save_freq=50_000,
    save_path=os.path.join(SAVE_DIR, "ckpts"),
    name_prefix="ppo_lander_cont",
)

model.learn(total_timesteps=800_000, callback=[eval_cb, ckpt_cb])
model.save(os.path.join(SAVE_DIR, "final"))

# quick headless eval of final and best models
mean_r_final, std_r_final = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
print(f"Eval over 20 episodes (final model): mean={mean_r_final:.1f} ± {std_r_final:.1f}")

model = PPO.load("ppo_lander_cont/best/best_model")
mean_r_best, std_r_best = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
print(f"Eval over 20 episodes (best model): mean={mean_r_best:.1f} ± {std_r_best:.1f}")

results_path = os.path.join(SAVE_DIR, "results.txt")

with open(results_path, "a") as f:
    f.write(f"Final model: mean={mean_r_final:.1f} ± {std_r_final:.1f}\n")
    f.write(f"Best model: mean={mean_r_best:.1f} ± {std_r_best:.1f}\n\n")

if mean_r_final > 200 or mean_r_best > 200:
    print("✅ Solved!")
