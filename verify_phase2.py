from rlstudio.agent import Agent
import torch


def test_agent_api():
    print("Testing Agent API...")
    agent = Agent(algorithm="PPO", env_id="CartPole-v1")

    print("Fitting agent...")
    agent.fit(max_epochs=2)

    print("Predicting...")
    obs = [0.0, 0.1, 0.0, 0.1]
    action = agent.predict(obs)
    print(f"Action: {action}")
    assert isinstance(action, torch.Tensor)
    print("Agent API Verified!")


if __name__ == "__main__":
    test_agent_api()
