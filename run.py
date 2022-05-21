import numpy as np
import yaml
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment

from train import train_ddpg

if __name__ == "__main__":

    env = UnityEnvironment(
        "/home/mustapha/Desktop/udacity_nano_degree/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux/Reacher.x86_64"
    )
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents in the environment
    print("Number of agents:", len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print("Number of actions:", action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)
    print("States have shape:", state.shape)

    agent_config = {
        "state_size": state_size,
        "action_size": action_size,
        "random_seed": 0,
    }
    scores = train_ddpg(
        env=env,
        brain=brain_name,
        agent_config=agent_config,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.show()
