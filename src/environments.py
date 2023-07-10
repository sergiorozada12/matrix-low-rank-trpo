import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv
from gym.envs.classic_control.acrobot import AcrobotEnv, wrap, bound, rk4


class CustomPendulumEnv(PendulumEnv):
    def reset(self):
        self.state = [np.random.rand()/100, np.random.rand()/100]
        self.last_u = None
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot])


class CustomAcrobotEnv(AcrobotEnv):
    def step(self, a):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        torque = np.clip(a, -1, 1)

        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(
                -self.torque_noise_max, self.torque_noise_max
            )

        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])

        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminated = self._terminal()
        reward = -1.0 if not terminated else 100.0

        return (self._get_ob(), reward, terminated, False, {})

    def _get_ob(self):
        s = self.state
        return np.array([s[0], s[1], s[2], s[3]])
