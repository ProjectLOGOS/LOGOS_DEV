import gym
from core.logos_validator_hub import validator_gate

class SimulatorClient:
    """
    Wraps OpenAI Gym environments for data collection.
    """
    def __init__(self, env_name: str):
        self.env = gym.make(env_name)

    @validator_gate
    def reset(self):
        """Reset environment and return initial state."""
        return self.env.reset()

    @validator_gate
    def step(self, action):
        """Perform action and return (state, reward, done, info)."""
        return self.env.step(action)
