# coding:utf-8
"""
Classic acrobot system implemented by Mahindrakar and Banavar
In part, Copied from OpenAI gym
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class AcrobotEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    grav = 9.8
    dt = 0.02

    LINK_LENGTH_1 = 1.0  # [m]
    LINK_LENGTH_2 = 2.0  # [m]
    LMAX = LINK_LENGTH_1 + LINK_LENGTH_2 + 0.2  # for display
    LINK_MASS_1 = 1.0  #: [kg] mass of link 1
    LINK_MASS_2 = 2.0  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 1.0  #: [m] position of the center of mass of link 2
    LINK_MOI_1 = 0.083  #: moments of inertia for link 1
    LINK_MOI_2 = 0.667  #: moments of inertia for link 2

    MAX_VEL_1 = 4.0 * np.pi
    MAX_VEL_2 = 8.0 * np.pi
    MAX_TORQUE = 5.0

    def __init__(self):
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])
        self.action_space = spaces.Box(low=-self.MAX_TORQUE, high=self.MAX_TORQUE, shape=(1,))
        self.observation_space = spaces.Box(-high, high)
        self._seed()
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        return self._get_obs()

    def _step(self, action):
        s = self.state
        torque = np.clip(action[0], -self.MAX_TORQUE, self.MAX_TORQUE)

        # Now, augment the state with our force action so it can be passed to
        ns = self._dynamics(s, torque, self.dt)   # more fast at the expense of accuracy

        ns[0] = angle_normalize(ns[0])
        ns[1] = angle_normalize(ns[1])
        ns[2] = np.clip(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = np.clip(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns

        # reward design
        reward = 0.0
        if len(action) == 4:
            done = False
            # reward -= action[1] * ( np.absolute(torque) / self.MAX_TORQUE - 0.5 ) * 2.0
            reward -= action[1] * ( ( np.absolute(torque) / self.MAX_TORQUE - 0.5 ) + ( np.absolute(ns[3]) / self.MAX_VEL_2 - 0.5 ) )
            reward += action[2] * ( - self.LINK_LENGTH_1 * np.cos(ns[0]) - self.LINK_LENGTH_2 * np.cos( ns[0] + ns[1] ) ) / ( self.LINK_LENGTH_1 + self.LINK_LENGTH_2 )
            reward += action[3] * ( np.absolute(ns[2]) / self.MAX_VEL_1 - 0.5 ) * 2.0
        else:
            done = self._terminal()
            reward = -1.0 if not done else 0.0

        return (self._get_obs(), reward, done, {})

    def _get_obs(self):
        s = self.state
        return np.array([np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]])

    def _terminal(self):
        s = self.state
        return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)

    def _dynamics(self, s, a, dt):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI_1
        I2 = self.LINK_MOI_2
        g = self.grav
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]

        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - 0.5 * np.pi)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - 0.5 * np.pi) + phi2
        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
            / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        return s + np.array([dtheta1, dtheta2, ddtheta1, ddtheta2]) * dt

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-self.LMAX,self.LMAX,-self.LMAX,self.LMAX)

        p1 = [self.LINK_LENGTH_1 * np.sin(s[0]),
              -self.LINK_LENGTH_1 *np.cos(s[0])]

        p2 = [p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1]),
              p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1])]

        xys = [[0,0], p1, p2]
        thetas = [s[0]-0.5*np.pi, s[0]+s[1]-0.5*np.pi]
        links = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-self.LMAX, 0), (self.LMAX, 0))
        for ((x,y),th,link) in zip(xys, thetas, links):
            l,r,t,b = 0, link, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
