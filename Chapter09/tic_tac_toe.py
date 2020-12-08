from gym.spaces import MultiDiscrete, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np


class TicTacToe(MultiAgentEnv):
    def __init__(self, config=None):
        self.s = 9
        self.action_space = Discrete(self.s)
        self.observation_space = MultiDiscrete([3] * self.s)
        self.agents = ["X", "O"]
        self.empty = " "
        self.t, self.state, self.rewards_to_send = self._reset()

    def _agent_observation(self, agent):
        obs = np.array([0] * self.s)
        for i, e in enumerate(self.state["board"]):
            if e == agent:
                obs[i] = 1
            elif e == self.empty:
                pass
            else:
                obs[i] = 2
        return obs

    def _next_agent(self, t):
        return self.agents[int(t % len(self.agents))]

    def _reset(self):
        t = 0
        agent = self._next_agent(t)
        state = {"turn": agent, "board": [self.empty] * self.s}
        rews = {a: 0 for a in self.agents}
        return t, state, rews

    def _if_won(self):
        rows = [
            (1, 2, 3),
            (4, 5, 6),
            (7, 8, 9),
            (1, 4, 7),
            (2, 5, 8),
            (3, 6, 9),
            (1, 5, 9),
            (3, 5, 7),
        ]
        if self.t >= 5:
            b = self.state["board"]
            for r1, r2, r3 in rows:
                if b[r1 - 1] == b[r2 - 1] == b[r3 - 1] == self.state["turn"]:
                    return True
        return False

    def _check_terminal(self):
        if self.t >= 9:
            return True
        elif self._if_won():
            return True
        return False

    def render(self):
        b = self.state["board"]
        d = "|"
        line = "-+-+-"
        print(b[0] + d + b[1] + d + b[2])
        print(line)
        print(b[3] + d + b[4] + d + b[5])
        print(line)
        print(b[6] + d + b[7] + d + b[8])

    def reset(self):
        self.t, self.state, self.rewards_to_send = self._reset()
        obs = {self.state["turn"]: self._agent_observation(self.state["turn"])}
        return obs

    def step(self, actions):
        assert len(actions) == 1, "Enter an action for a single player"
        agent = list(actions)[0]
        assert agent == self.state["turn"]

        action = actions[agent]
        if self.state["board"][action] == self.empty:
            self.state["board"][action] = agent
            self.t += 1
            done = self._check_terminal()
            if done:
                if self._if_won():
                    for a in self.rewards_to_send:
                        if a == agent:
                            self.rewards_to_send[a] = 1
                        else:
                            self.rewards_to_send[a] = -1
                else:
                    self.rewards_to_send[agent] = 0
            else:
                self.rewards_to_send[agent] = 0
                self.state["turn"] = self._next_agent(self.t)
        else:
            done = False
            self.rewards_to_send[agent] = -10

        assert self.t < 10
        if done:
            obs = {a: self._agent_observation(a) for a in self.agents}
            rewards = {a: self.rewards_to_send[a] for a in self.agents}
        else:
            next_agent = self.state["turn"]
            obs = {next_agent: self._agent_observation(next_agent)}
            rewards = {next_agent: self.rewards_to_send[next_agent]}
        dones = {"__all__": done}
        infos = {}

        return obs, rewards, dones, infos
