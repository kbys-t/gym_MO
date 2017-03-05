"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
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
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.dt = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.x_dot_max = 1.0 * self.x_threshold
        self.theta_dot_max = 4.0 * np.pi

        high = np.array([
            self.x_threshold,
            self.x_dot_max,
            1.0,
            1.0,
            self.theta_dot_max])

        self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = np.clip(action[0], -self.force_mag, self.force_mag)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  += self.dt * x_dot
        x_dot += self.dt * xacc
        x_dot = np.clip(x_dot, -self.x_dot_max, self.x_dot_max)
        theta += self.dt * theta_dot
        theta_dot += self.dt * thetaacc
        theta = angle_normalize(theta)
        theta_dot = np.clip(theta_dot, -self.theta_dot_max, self.theta_dot_max)
        self.state = (x,x_dot,theta,theta_dot)

        done =  x < -self.x_threshold \
                or x > self.x_threshold
        done = bool(done)

        # reward design
        reward = 0.0
        if len(action) == 4:
            reward -= action[1] * ( ( np.absolute(force) / self.force_mag - 0.5 ) + ( np.absolute(x) / self.x_threshold - 0.5 ) )
            reward += action[2] * np.cos(theta)
            reward += action[3] * ( np.absolute(theta_dot) / self.theta_dot_max - 0.5 ) * 2.0
            reward = -1.0 if done else reward
        else:
            reward = 1.0 if np.absolute(x) < self.x_threshold and np.absolute(theta) < self.theta_threshold_radians else 0.0

        return np.array([x,x_dot,np.cos(theta),np.sin(theta),theta_dot]), reward, done, {}

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        # self.state[2] += np.pi
        x, x_dot, theta, theta_dot = self.state
        return np.array([x,x_dot,np.cos(theta),np.sin(theta),theta_dot])

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 500
        screen_height = 500

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 250 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
