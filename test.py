import gymnasium as gym
from stable_baselines3 import PPO

# simple visualization of the best model
env = gym.make("LunarLander-v3", continuous = True, render_mode="human")
model = PPO.load("ppo_lander_cont/best/best_model")
obs, _ = env.reset(seed=0)
done = truncated = False
while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, truncated, _ = env.step(action)
env.close() 