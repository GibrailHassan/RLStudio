import argparse
import sys
from rlstudio import Agent


def main():
    parser = argparse.ArgumentParser(description="RLStudio Training CLI")
    parser.add_argument("--algo", type=str, default="PPO", help="Algorithm (PPO, DQN)")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment ID")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    args = parser.parse_args()

    print(f"Starting Training: {args.algo} on {args.env} for {args.epochs} epochs.")

    agent = Agent(algorithm=args.algo, env_id=args.env)
    agent.fit(max_epochs=args.epochs)

    print("Training Complete.")


if __name__ == "__main__":
    main()
