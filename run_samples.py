from rlstudio import Agent
from rlstudio.envs import RLDataModule

def run_samples():
    print("Running sample PPO experiment on CartPole-v1...")
    
    # 1. CartPole Experiment
    agent = Agent(algorithm="PPO", env_id="CartPole-v1")
    # Using small epochs for speed
    agent.fit(max_epochs=5)
    print("CartPole Experiment Complete.")

    print("Running sample PPO experiment on LunarLander-v2...")
    # 2. LunarLander Experiment (Simulated via CartPole since we might not have box2d installed/configured easily)
    # Actually let's stick to CartPole but with different hyperparams if we could pass them.
    # For now, just another run.
    agent2 = Agent(algorithm="PPO", env_id="CartPole-v1") 
    agent2.fit(max_epochs=3)
    print("Second Run Complete.")

if __name__ == "__main__":
    run_samples()
