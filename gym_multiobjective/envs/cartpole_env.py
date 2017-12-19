# coding:utf-8
"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
In part, Modified from OpenAI gym
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # Constants
        self.GRAVITY = 9.8
        self.DT = 0.02
        # Physical params
        self.MASS_CART = 1.0
        self.MASS_POLE = 0.1
        self.MASS_TOTAL = (self.MASS_POLE + self.MASS_CART)
        self.LEN_POLE = 0.5 # actually half the pole's length
        self.MASSLEN_POLE = (self.MASS_POLE * self.LEN_POLE)
        # Limitation
        self.MAX_ANG = 12 * 2 * np.pi / 360   # not use in MO
        self.MAX_X = 2.4
        self.MAX_VEL_X = 5.0 * self.MAX_X
        self.MAX_VEL_ANG = 4.0 * np.pi
        self.MAX_FORCE = 10.0

        # Create spaces
        high = np.array([self.MAX_X, 1.0, 1.0, self.MAX_VEL_X, self.MAX_VEL_ANG])
        self.action_space = spaces.Box(low=-self.MAX_FORCE, high=self.MAX_FORCE, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

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
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        # self.state[1] += np.pi
        return self._get_obs()

    def _step(self, action):
        s = self.state
        force = np.clip(action[0], -self.MAX_FORCE, self.MAX_FORCE)

        ns = self._dynamics(s, force, self.DT)

        ns[2] = np.clip(ns[2], -self.MAX_VEL_X, self.MAX_VEL_X)
        ns[3] = np.clip(ns[3], -self.MAX_VEL_ANG, self.MAX_VEL_ANG)
        ns[1] = angle_normalize(ns[1])
        collision = np.absolute(ns[0]) > self.MAX_X
        if collision:
            ns[0] = np.copysign(self.MAX_X, ns[0])
            ns[2] = 0.0

        self.state = ns

        # reward design
        reward = 0.0
        if len(action) == self.action_space.shape[0] + self.TASK_NUM:
            done = False
            if collision:
                reward = -1.0
            else:
                reward += action[1] * np.cos(ns[1])
                reward += action[2] * ( np.absolute(ns[3]) / self.MAX_VEL_ANG - 0.5 ) * 2.0
                reward -= action[3] * ( ( np.absolute(force) / self.MAX_FORCE - 0.5 ) + ( np.absolute(ns[0]) / self.MAX_X - 0.5 ) )
        else:
            done = collision or np.absolute(ns[1]) > self.MAX_ANG
            reward = 0.0 if done else 1.0

        return (self._get_obs(), reward, done, {})

    def _get_obs(self):
        s = self.state
        return np.array([s[0], np.cos(s[1]), np.sin(s[1]), s[2], s[3]])

    def _dynamics(self, s, a, dt):
        x, theta, dx, dtheta = s

        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (a + self.MASSLEN_POLE * dtheta * dtheta * sintheta) / self.MASS_TOTAL
        ddtheta = (self.GRAVITY * sintheta - costheta * temp) / (self.LEN_POLE * (4.0/3.0 - self.MASS_POLE * costheta * costheta / self.MASS_TOTAL))
        ddx  = temp - self.MASSLEN_POLE * ddtheta * costheta / self.MASS_TOTAL

        return s + np.array([dx, dtheta, ddx, ddtheta]) * dt

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 500
        screen_height = 500

        world_width = self.MAX_X*2
        scale = screen_width/world_width
        carty = 250 # TOP OF CART
        polewidth = 10.0 * 0.5
        polelen = scale * 1.0
        cartwidth = 50.0 * 0.5
        cartheight = 30.0 * 0.5

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth, cartwidth, cartheight, -cartheight
            axleoffset =cartheight * 0.5
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth, polewidth, polelen - polewidth, -polewidth
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(self.track)

        s = self.state
        cartx = s[0]*scale + 0.5*screen_width # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-s[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
