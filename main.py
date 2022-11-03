import os
import gym
import pickle
import argparse
from datetime import datetime
from project.agents.core import BaseAgent
from project.agents.q_learning_agent import QLearningAgent
from project.agents.dyna_q_agent import DynaQAgent
from project.enviroment import Enviroment


parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=200)
parser.add_argument('--agent', type=str, default="QLearning")
parser.add_argument('--render_steps', type=int, default=50)


def save_results(agent_type, q_values, steps, n_episodes):
    result_folder = 'saves/results'
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    now = datetime.now().strftime("%m%d%Y_%H%M%S")
    results_path = os.path.join(result_folder,
                                f'{agent_type}_{n_episodes}_{now}.pickle')
    results = {'agent_type': agent_type,
               'q_values': q_values,
               'steps': steps,
               'episodes': n_episodes}
    with open(results_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def episode(agent: BaseAgent, env: gym.Env, render: bool = False):
    observation, info = env.reset()
    action = agent.agent_start(observation)
    if render:
        env.render()
    terminated = False
    steps = 0
    while not terminated:
        steps += 1
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            agent.agent_end(reward)
            observation, info = env.reset()
            break
        action = agent.agent_step(observation, reward)
    return steps


def main():
    args = parser.parse_args()
    agent_type = args.agent
    episodes = args.episodes
    render_steps = args.render_steps
    saves_folder = 'saves'
    if not os.path.exists(saves_folder):
        os.mkdir(saves_folder)
    map = Enviroment()
    if agent_type == "QLearning":
        agent = QLearningAgent()
    else:
        agent = DynaQAgent()
    agent_info = {"num_actions": 4,
                  "num_states": 16,
                  "epsilon": 0.1,
                  "step_size": 0.1,
                  "discount": 1.0}
    agent.agent_init(agent_info)
    n_steps = []
    for i in range(episodes):
        render_bool = i % render_steps
        steps = episode(agent, map.env, render_bool)
        print("Episode {} finished after {} steps".format(i+1, steps+1))
        n_steps.append(steps)
    agent.agent_save()
    save_results(agent_type, agent.q_values, n_steps, episodes)
    map.env.close()


if __name__ == "__main__":
    main()