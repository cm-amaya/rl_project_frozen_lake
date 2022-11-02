import gym
import argparse
from agent import Agent
from enviroment import Enviroment


parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=200)
parser.add_argument('--agent', type=str, default="Base")


def episode(agent: Agent, env: gym.Env):
    observation, info = env.reset()
    action = agent.agent_start(observation)
    env.render()
    terminated = False
    steps = 0
    while not terminated:
        steps += 1
        observation, reward, terminated, truncated, info = env.step(action)
        print(f'Act: {action} Obs: {observation} Reward:{reward} Inf:{info}')
        if terminated or truncated:
            agent.agent_end(reward)
            observation, info = env.reset()
            print("Episode finished after {} steps".format(steps+1))
            break
        action = agent.agent_step(observation, reward)
    return steps

def main():
    map = Enviroment()
    agent_info = {"num_actions": 4,
                  "num_states": 16,
                  "epsilon": 0.1,
                  "step_size": 0.1,
                  "discount": 1.0}
    agent = Agent(agent_info)
    episodes = 500
    n_steps = []
    for _ in range(episodes):
        steps = episode(agent, map.env)
        n_steps.append(steps)
    map.env.close()


if __name__ == "__main__":
    main()