import yaml
from unityagents import UnityEnvironment

if __name__ == "__main__":

    env = UnityEnvironment(
        "/home/mustapha/Desktop/udacity_nano_degree/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux/Reacher.x86_64"
    )

    with open("config.yaml", "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    agent_hyperparams = params["hyperparameters"]
    training_params = params["trainingparameters"]
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
