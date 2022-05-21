from collections import deque

import numpy as np
import torch
import yaml

from ddpg_agent import Agent


def train_ddpg(
    env,
    brain,
    agent_config,
    n_episodes=2000,
    max_t=1000,
    time_steps=20,
    update=10,
):
    """Deep Q-Learning.

    Params
    ======
        env : environment
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        time_steps (int) :
    """
    env_info = env.reset(train_mode=True)[brain]

    avg_score = []
    scores_deque = deque(maxlen=100)
    scores = np.zeros(len(env_info.agents))

    env_info = env.reset(train_mode=True)[brain]

    states = env_info.vector_observations

    agents = [Agent(**agent_config) for _ in range(len(env_info.agents))]
    action = [agent.act(states[i]) for i, agent in enumerate(agents)]
    for i_episode in range(1, n_episodes + 1):
        state = env_info.vector_observations
        for agent in agents:
            agent.reset()
        for t in range(max_t):
            actions = [agent.act(states[i]) for i, agent in enumerate(agents)]
            env_info = env.step(actions)[brain]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            step_t = zip(agents, states, actions, rewards, next_states, dones)

            for agent, state, action, reward, next_step, done in step_t:
                agent.memory.add(state, action, reward, next_step, done)
                if t % time_steps == 0:
                    agent.step(state, action, reward, next_step, done, update)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        score = np.mean(scores)
        avg_score.append(score)
        scores_deque.append(score)
        avg = np.mean(scores_deque)

        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(
                i_episode,
                avg,
            ),
            end="\n",
        )

        if np.mean(scores_deque) > 30.0:
            print(
                f"Enviroment solved in episode={i_episode} avg_score={avg:.2f}".format(
                    i_episode=i_episode, avg=avg
                )
            )

            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")
            break
    return avg_score
