"""
Based on the source of OpenAI gym
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class PendulumSwingEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.GRAVITY = 9.8
        self.DT = 0.02
        self.LINK_LENGTH = 1.0
        self.LINK_MASS = 1.0

        self.MAX_VEL = 2.0 * np.pi
        self.MAX_TORQUE = 3.0

        high = np.array([1.0, 1.0, self.MAX_VEL])
        self.action_space = spaces.Box(low=-self.MAX_TORQUE, high=self.MAX_TORQUE, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        # set the number of tasks
        self.TASK_NUM = 3
        self.TASK_NAME = ["height", "velocity", "energy"]

        # Initialize
        self._seed()
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        # high = np.array([np.pi, 1])
        high = np.array([0.05, 0.05])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.state[0] += np.pi  # difference is here
        self.last_u = None
        return self._get_obs()

    def _step(self, action):
        s = self.state
        torque = np.clip(action[0], -self.MAX_TORQUE, self.MAX_TORQUE)
        self.last_u = torque # for rendering

        ns = self._dynamics(s, torque, self.DT)

        ns[1] = np.clip(ns[1], -self.MAX_VEL, self.MAX_VEL)
        ns[0] = angle_normalize(ns[0])

        self.state = ns

        # reward design
        reward = 0.0
        done = False
        if len(action) == self.action_space.shape[0] + self.TASK_NUM:
            reward += action[1] * np.cos(ns[0])
            reward += action[2] * ( np.absolute(ns[1]) / self.MAX_VEL - 0.5 ) * 2.0
            reward -= action[3] * ( ( np.absolute(torque) / self.MAX_TORQUE - 0.5 ) * 2.0 )
        else:
            reward = - (s[0] + 0.1*s[1]**2 + .001*(torque**2))

        return (self._get_obs(), reward, done, {})

    def _get_obs(self):
        s = self.state
        return np.array([np.cos(s[0]), np.sin(s[0]), s[1]])

    def _dynamics(self, s, a, dt):
        m = self.LINK_MASS
        l = self.LINK_LENGTH
        g = self.GRAVITY
        theta, dtheta = s

        ddtheta = (-3.0*g/(2.0*l) * np.sin(theta + np.pi) + 3.0/(m*l**2)*a)

        return s + np.array([dtheta, ddtheta]) * dt

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(self.LINK_LENGTH, 0.2*self.LINK_LENGTH)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05*self.LINK_LENGTH)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + 0.5*np.pi)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/self.MAX_TORQUE, np.abs(self.last_u)/self.MAX_TORQUE)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
