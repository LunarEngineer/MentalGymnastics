# import gym
import random
import numpy as np
import torch
import mentalgym
import mentalgym.envs
import mentalgym.functionbank
import agentNets
from collections import deque

max_steps = 10
num_episodes = 2
num_active_fns_init = 3
epsilon_start = 1.0
alpha_start = 0.001
gamma = 0.99
buffer_len = 100
num_functions = 8
minibatch_size = 8
min_buffer_use_size = 10 * minibatch_size


class MentalAgent:
    def __init__(self):
        # Initialize RL Hyperparameters
        self.epsilon = epsilon_start
        self.alpha = alpha_start
        self.gamma = gamma

        # Instantiate environment
        self.env = mentalgym.envs.MentalEnv()

        # Create DQN NN
        self.DQN_agent_Q = agentNets.DQNAgentNN()

        # Instantiate loss criterion
        self.criterion = torch.nn.MSELoss()

        # Instantiate Optimizer
        self.optimizer = torch.optim.Adam(
            self.DQN_agent_Q.parameters(), lr=self.alpha
        )

        # Mask for invalid functions
        self.num_active_fns = num_active_fns_init

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_len)

        # Variable to store all the episodic rewards
        self.rewards = np.zeros(num_episodes)

    def train(self):
        # Iterate over episodes
        for e in range(num_episodes):
            # Initialize state
            S = self.env.reset()

            # Accumulate reward for each episode
            dqn_reward = 0

            # Count time steps for each episode
            t = 0

            # Flag to detect episode termination
            done = False

            # Iterate over time steps in an episode
            while not done:
                t += 1

                # Vectorize state by changing function IDs into one-hot vectors
                fns_oh = np.zeros((max_steps, num_functions))
                for i in range(max_steps):
                    fns_oh[
                        np.arange(max_steps),
                        S["experiment_space"]["function_ids"][i],
                    ] = 1

                dqn_S = torch.cat(
                    (
                        torch.Tensor(fns_oh.flatten()),
                        torch.Tensor(
                            S["experiment_space"]["function_locations"]
                        ),
                        torch.Tensor(
                            [S["experiment_space"]["function_connection"]]
                        ),
                    )
                )

                # Obtain action values
                with torch.no_grad():
                    dqas = self.DQN_agent_Q(dqn_S.reshape(1, len(dqn_S)))

                # Behavior Policy: epsilon-greedy
                if np.random.rand() < self.epsilon:
                    dqn_action = np.random.randint(num_active_fns_init)
                else:
                    dqn_action = np.argmax(dqas[0])

                # TODO: need to supply location and radius components of action
                action = {
                    "function_id": dqn_action,
                    "location": (0, 0),
                    "radius": 0,
                }

                # Take a step in the environment
                Sp, dqn_R, done, info = self.env.step(action)

                # Accumulate episodic reward
                dqn_reward += dqn_R

                # Vectorize next state by changing function IDs into
                # one-hot vectors
                fnsp_oh = np.zeros((max_steps, num_functions))
                for i in range(max_steps):
                    fnsp_oh[
                        np.arange(max_steps),
                        Sp["experiment_space"]["function_ids"][i],
                    ] = 1

                dqn_Sp = np.concatenate(
                    (
                        fnsp_oh.flatten(),
                        Sp["experiment_space"]["function_locations"],
                        [Sp["experiment_space"]["function_connection"]],
                    )
                )

                # Append experience to replay buffer
                self.replay_buffer.append(
                    (dqn_S.numpy(), dqn_action, dqn_R, dqn_Sp, done)
                )

                # Sample minibatch from buffer if buffer is filled enough
                if len(self.replay_buffer) > min_buffer_use_size:
                    minibatch = np.array(
                        random.sample(self.replay_buffer, minibatch_size),
                        dtype=object,
                    )
                    Xr = torch.Tensor(
                        np.stack(np.concatenate(minibatch[:, 0:1]))
                    )
                    Ar = torch.tensor(
                        np.array(minibatch[:, 1:2], dtype=int),
                        dtype=torch.long,
                    )
                    Rr = torch.Tensor(np.array(minibatch[:, 2:3], dtype=float))
                    Spr = torch.Tensor(
                        np.stack(np.concatenate(np.array(minibatch[:, 3:4])))
                    )
                    doner = torch.tensor(
                        np.array(minibatch[:, 4:5], dtype=int),
                        dtype=torch.int32,
                    )

                    # Train DQN NN
                    self.optimizer.zero_grad()
                    target = (
                        Rr
                        + self.gamma
                        * torch.max(self.DQN_agent_Q(Spr), 1, keepdim=True)[0]
                    )
                    target = (1 - doner) * target + doner * Rr
                    Y = self.DQN_agent_Q(Xr)
                    Yt = Y.clone()
                    Yt[torch.arange(0, minibatch_size), Ar] = target
                    loss = self.criterion(Y, Yt)
                    loss.backward()
                    self.optimizer.step()

                # Advance state
                S = Sp

                # Print Status
                print("Episode:", e + 1, "Time step:", t)


if __name__ == "__main__":
    agent = MentalAgent()
    agent.train()

    function_bank = mentalgym.functionbank.FunctionBank("mentalgym/functions")
#    print(function_bank)
