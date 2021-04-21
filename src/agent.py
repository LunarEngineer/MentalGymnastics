# import gym
import random
import numpy as np
import torch
import mentalgym
import mentalgym.envs
import mentalgym.functionbank
import agentNets
from collections import deque


class MentalAgent:
    def __init__(self, hparams):
        # Initialize RL Hyperparameters
        self.epsilon = hparams["epsilon_start"]
        self.alpha = hparams["alpha_start"]

        # Instantiate environment
        self.env = mentalgym.envs.MentalEnv()

        # Create DQN NN
        self.DQN_agent_Q = agentNets.DQNAgentNN(hparams)

        # Instantiate loss criterion
        self.criterion = torch.nn.MSELoss()

        # Instantiate Optimizer
        self.optimizer = torch.optim.Adam(
            self.DQN_agent_Q.parameters(), lr=self.alpha
        )

        # Mask for invalid functions
        self.num_active_fns = hparams["num_active_fns_init"]

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=hparams["buffer_len"])

        # Variable to store all the episodic rewards
        self.rewards = np.zeros(hparams["num_episodes"])

    def train(self, hparams):
        # Iterate over episodes
        for e in range(hparams["num_episodes"]):
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
                fns_oh = np.zeros(
                    (hparams["max_steps"], hparams["num_functions"])
                )
                for i in range(hparams["max_steps"]):
                    fns_oh[
                        np.arange(hparams["max_steps"]),
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
                    dqn_action = np.random.randint(
                        hparams["num_active_fns_init"]
                    )
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
                fnsp_oh = np.zeros(
                    (hparams["max_steps"], hparams["num_functions"])
                )
                for i in range(hparams["max_steps"]):
                    fnsp_oh[
                        np.arange(hparams["max_steps"]),
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
                if len(self.replay_buffer) > hparams["min_buffer_use_size"]:
                    minibatch = np.array(
                        random.sample(
                            self.replay_buffer, hparams["minibatch_size"]
                        ),
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
                        + hparams["gamma"]
                        * torch.max(self.DQN_agent_Q(Spr), 1, keepdim=True)[0]
                    )
                    target = (1 - doner) * target + doner * Rr
                    Y = self.DQN_agent_Q(Xr)
                    Yt = Y.clone()
                    Yt[torch.arange(0, hparams["minibatch_size"]), Ar] = target
                    loss = self.criterion(Y, Yt)
                    loss.backward()
                    self.optimizer.step()

                # Advance state
                S = Sp

                # Print Status
                print("Episode:", e + 1, "Time step:", t)


if __name__ == "__main__":
    # Customize training run **HERE**
    hparams = {}
    hparams["num_episodes"] = 10
    hparams["max_steps"] = 10
    hparams["hidden_layers"] = (10,)
    hparams["gamma"] = 0.99
    hparams["alpha_start"] = 0.001
    hparams["alpha_const"] = 2.0
    hparams["alpha_maintain"] = 0.00001
    hparams["epsilon_start"] = 1.0
    hparams["epsilon_const"] = 20.0
    hparams["epsilon_maintain"] = 0.01
    hparams["buffer_len"] = 100
    hparams["minibatch_size"] = 8
    hparams["min_buffer_use_size"] = 5 * hparams["minibatch_size"]
    hparams["num_functions"] = 8
    hparams["num_active_fns_init"] = 3

    agent = MentalAgent(hparams)
    agent.train(hparams)

#    function_bank = mentalgym.functionbank.FunctionBank("mentalgym/functions")
#    print(function_bank)
