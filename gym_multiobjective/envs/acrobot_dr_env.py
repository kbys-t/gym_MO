# coding:utf-8
"""
Classic acrobot system implemented based on Brown and Passino, 1997.
Friction is added according to Yoshimoto et al., 1999.
In part, Copied from OpenAI gym
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class AcrobotDREnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # Constants
        self.GRAVITY = 9.8
        self.DT = 0.02
        # Physical params
        self.LINK_LENGTH_1 = 0.2  # [m]
        self.LINK_LENGTH_2 = 0.2  # [m]
        self.LINK_MASS_1 = 1.9008  #: [kg] mass of link 1
        self.LINK_MASS_2 = 0.7175  #: [kg] mass of link 2
        self.LINK_COM_POS_1 = 1.8522e-1  #: [m] position of the center of mass of link 1
        self.LINK_COM_POS_2 = 6.2052e-2  #: [m] position of the center of mass of link 2
        self.LINK_MOI_1 = 4.3399e-3  #: moments of inertia for link 1
        self.LINK_MOI_2 = 5.2285e-3  #: moments of inertia for link 2
        self.FRICTION = 0.0    #: friction coefficent for both joints. If you set, 0.01 is the same parameter as Yoshimoto et al.
        self.LMAX = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.05  # for display
        # Limitation
        self.MAX_VEL_1 = 4.0 * np.pi
        self.MAX_VEL_2 = 4.0 * np.pi
        self.MAX_TORQUE = 2.5
        self.MAX_ANG_2 = np.pi

        # Create spaces
        high = np.array([1.0, 1.0, self.MAX_ANG_2, self.MAX_VEL_1, self.MAX_VEL_2])
        self.action_space = spaces.Box(low=-self.MAX_TORQUE, high=self.MAX_TORQUE, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

        # Initialize
        self._seed()
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return self._get_obs()

    def _step(self, action):
        s = self.state
        torque = np.clip(action[0], -self.MAX_TORQUE, self.MAX_TORQUE)

        ns = self._dynamics(s, torque, self.DT)

        ns[2] = np.clip(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = np.clip(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        ns[0] = angle_normalize(ns[0])
        collision =np.absolute(ns[1]) > self.MAX_ANG_2
        if collision:
            ns[1] = np.clip(ns[1], -self.MAX_ANG_2, self.MAX_ANG_2)
            ns[3] = 0.0

        self.state = ns

        # reward design
        reward = 0.0
        done = False
        if len(action) == 4:
            if collision:
                reward = -1.0
            reward += (1.0 - np.power(torque / self.MAX_TORQUE, 2)) * 0.02
            if np.argmax(action[1:]) == 1:
                # height
                done = ( - self.LINK_LENGTH_1 * np.cos(ns[0]) - self.LINK_LENGTH_2 * np.cos( ns[0] + ns[1] ) ) / ( self.LINK_LENGTH_1 + self.LINK_LENGTH_2 ) > 0.95
                reward += 100.0 if done else 0.0
            elif np.argmax(action[1:]) == 2:
                # velocity
                done = np.absolute(ns[2]) / self.MAX_VEL_1 > 0.95
                reward += 100.0 if done else 0.0
        else:
            done = ( - self.LINK_LENGTH_1 * np.cos(ns[0]) - self.LINK_LENGTH_2 * np.cos( ns[0] + ns[1] ) ) / ( self.LINK_LENGTH_1 + self.LINK_LENGTH_2 ) > 0.5
            reward = 0.0 if done else -1.0

        return (self._get_obs(), reward, done, {})

    def _get_obs(self):
        s = self.state
        # return np.array([np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]])
        return np.array([np.cos(s[0]), np.sin(s[0]), s[1], s[2], s[3]])

    def _dynamics(self, s, a, dt):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI_1
        I2 = self.LINK_MOI_2
        mu = self.FRICTION
        g = self.GRAVITY
        theta1, theta2, dtheta1, dtheta2 = s

        # coeffs
        d22 = m2 * lc2**2 + I2
        d11 = m1 *lc1**2 + m2 * (l1**2 + 2.0*l1*lc2*np.cos(theta2)) + I1 + d22
        d12 = m2 * l1*lc2*np.cos(theta2) + d22
        m2l1lc2st2 = m2 * l1*lc2*np.sin(theta2)
        hphi2 = m2*g * lc2*np.cos(theta1 + theta2 - 0.5*np.pi)
        hphi1 = - m2l1lc2st2 * ( dtheta2**2 + 2.0*dtheta1*dtheta2 ) \
              + (m1 * lc1 + m2 * l1)*g * np.cos(theta1 - 0.5*np.pi) + hphi2
        hphi2 += m2l1lc2st2 * dtheta1**2
        # acceleration
        ddtheta2 = ( d11*(a - mu*dtheta2) + d12*hphi1 - d11*hphi2 ) / ( d11*d22 - d12**2 )
        ddtheta1 = - ( d12*ddtheta2 + hphi1 + mu*dtheta1 ) / d11

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
            l,r,t,b,s = 0, link, 0.02, -0.02, 0.02
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0, 0.8, 0.8)
            circ = self.viewer.draw_circle(s)
            circ.set_color(0.8, 0.8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
