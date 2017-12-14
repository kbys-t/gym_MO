# coding:utf-8
"""
Two link arm for reaching or hitting object
In part, Copied from OpenAI gym
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class BallArmStaticEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # Constants
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
        self.LMAX = self.LINK_LENGTH_1 + self.LINK_LENGTH_2  # for display
        self.LINK_SIZE = 0.025
        self.OBJ_MASS = 0.0027
        self.OBJ_SPRING = self.OBJ_MASS * 5.0**2
        self.OBJ_DAMPER = 0.2 * 2.0 * np.sqrt(self.OBJ_MASS * self.OBJ_SPRING)
        self.OBJ_REST = np.array([0.0, -0.6*self.LMAX])
        self.OBJ_SIZE = 0.02
        
        # Limitation
        self.MAX_VEL = 10.0 * 0.75
        self.MAX_TORQUE = 4.5 * 0.5
        self.MAX_ANG = np.pi
        self.MAX_OBJ_POS = self.LMAX + 0.05
        self.MAX_OBJ_VEL = 10.0

        # Create spaces
        high_obs = np.array([self.MAX_ANG, self.MAX_ANG, self.MAX_VEL, self.MAX_VEL, self.MAX_OBJ_POS, self.MAX_OBJ_POS, self.MAX_OBJ_VEL, self.MAX_OBJ_VEL])
        high_act = np.array([self.MAX_TORQUE, self.MAX_TORQUE])
        self.observation_space = spaces.Box(-high_obs, high_obs)
        self.action_space = spaces.Box(-high_act, high_act)

        # set the number of tasks
        self.TASK_NUM = 4

        # Initialize
        self._seed()
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(8,))
        self.state[0] += np.pi
        self.state[4:6] += self.OBJ_REST
        return self._get_obs()

    def _step(self, action):
        s = self.state
        torque = np.clip(action[:2], -self.MAX_TORQUE, self.MAX_TORQUE)

        ns = self._dynamics(s, torque, self.DT)

        # limit of two link
        collision = np.absolute(ns[0:2]) > self.MAX_ANG
        ns[2:4][np.absolute(ns[0:2]) > self.MAX_ANG] = 0.0
        ns[0:2] = np.clip(ns[0:2], -self.MAX_ANG, self.MAX_ANG)
        ns[2:4] = np.clip(ns[2:4], -self.MAX_VEL, self.MAX_VEL)
        # limit of object
        ns[6:8][np.absolute(ns[4:6]) > self.MAX_OBJ_POS] *= -1.0
        ns[4:6] = np.clip(ns[4:6], -self.MAX_OBJ_POS+self.OBJ_SIZE, self.MAX_OBJ_POS-self.OBJ_SIZE)
        ns[6:8] = np.clip(ns[6:8], -self.MAX_OBJ_VEL, self.MAX_OBJ_VEL)

        self.state = ns

        # reward design
        reward = 0.0
        done = False
        if any(collision):
            reward = -1.0
        elif len(action) == 6:
            pt = np.array([ self.LINK_LENGTH_1 * np.sin(ns[0]) + self.LINK_LENGTH_2 * np.sin(ns[0] + ns[1]) , \
                - self.LINK_LENGTH_1 *np.cos(ns[0]) - self.LINK_LENGTH_2 * np.cos(ns[0] + ns[1]) ])
            reward -= action[2] * ( np.absolute(torque).mean() / self.MAX_TORQUE - 0.5 ) * 2.0
            reward += action[3] * ( np.exp( - 2.0 * np.linalg.norm(pt - ns[4:6]) ) - 0.5 ) * 2.0
            reward += action[4] * ( 0.5 - np.exp( - 2.0 * np.linalg.norm(ns[6:8]) ) ) * 2.0
            reward += action[5] * ( ns[5] / self.MAX_OBJ_POS )  # tentative
        else:
            done = np.linalg.norm(ns[4:6]) > self.LMAX + 2.0*self.OBJ_SIZE
            reward = 1.0 if done else -1.0

        return (self._get_obs(), reward, done, {})

    def _get_obs(self):
        s = self.state
        return s


    def _dynamics(self, s, a, dt):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        l2 = self.LINK_LENGTH_2
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI_1
        I2 = self.LINK_MOI_2
        mu = self.FRICTION
        mc = self.OBJ_MASS
        kc = self.OBJ_SPRING
        dc = self.OBJ_DAMPER
        pr = self.OBJ_REST
        sz = self.OBJ_SIZE + self.LINK_SIZE
        theta1, theta2, dtheta1, dtheta2, x, y, dx, dy = s

        # twolink
        # coeffs
        d22 = m2 * lc2**2 + I2
        d11 = m1 *lc1**2 + m2 * (l1**2 + 2.0*l1*lc2*np.cos(theta2)) + I1 + d22
        d12 = m2 * l1*lc2*np.cos(theta2) + d22
        m2l1lc2st2 = m2 * l1*lc2*np.sin(theta2)
        hphi1 = - m2l1lc2st2 * ( dtheta2**2 + 2.0*dtheta1*dtheta2 )
        hphi2 = m2l1lc2st2 * dtheta1**2
        # acceleration
        ddtheta2 = ( d11*(a[1] - mu*dtheta2) + d12*hphi1 - d11*hphi2 ) / ( d11*d22 - d12**2 )
        ddtheta1 = - ( d12*ddtheta2 + hphi1 + mu*dtheta1 - a[0] ) / d11

        # object
        ddx = - ( kc * (x-pr[0]) + dc * dx) / mc
        ddy = - ( kc * (y-pr[1]) + dc * dy) / mc

        # update
        ns = s + np.array([dtheta1, dtheta2, ddtheta1, ddtheta2, dx, dy, ddx, ddy]) * dt

        # collision
        # positions
        p1 = np.array([l1 * np.sin(ns[0]),  - l1 * np.cos(ns[0])])
        dp = np.array([l2 * np.sin(ns[0]+ns[1]), - l2 * np.cos(ns[0]+ns[1])])
        pc = np.array([ns[4], ns[5]])
        p2 = p1 + dp
        i1 = np.array([ p1[0]**2*pc[0] + p1[0]*p1[1]*pc[1] , p1[0]*p1[1]*pc[0] + p1[1]**2*pc[1] ]) \
            / ( p1[0]**2 + p1[1]**2 )
        i2 = np.array([ dp[0]**2*pc[0] + dp[1]**2*p1[0] + dp[0]*dp[1]*(pc[1] - p1[1]) , dp[0]*dp[1]*(pc[0] - p1[0]) + dp[1]**2*pc[1] + dp[0]**2*p1[1] ]) \
             / ( dp[0]**2 + dp[1]**2 )
        d1 = np.linalg.norm(pc-i1)
        d2 = np.linalg.norm(pc-i2)
        # judge
        istip = np.linalg.norm(pc-p2) <= self.OBJ_SIZE
        isl2 = d2 <= sz and i2[0] >= np.minimum(p1[0], p2[0]) and i2[0] <= np.maximum(p1[0], p2[0])
        isl1 = d1 <= sz and i1[0] >= np.minimum(0.0, p1[0]) and i1[0] <= np.maximum(0.0, p1[0])
        if isl2 or istip:
            # velocity update (arm velocity is assumed to be constant)
            ec = 1.0
            mg = m1 + m2
            pgn = np.array([\
                m1*( lc1 * np.sin(ns[0]) ) + m2*( p1[0] + lc2 * np.sin(ns[0]+ns[1]) ) , \
                m1*( - lc1 * np.cos(ns[0]) ) + m2*( p1[1] - lc2 * np.cos(ns[0]+ns[1]) ) \
                ]) / mg
            pgo = np.array([\
                m1*( lc1 * np.sin(s[0]) ) + m2*( l1 * np.sin(s[0]) + lc2 * np.sin(s[0]+s[1]) ) , \
                m1*( - lc1 * np.cos(s[0]) ) + m2*( - l1 * np.cos(s[0]) - lc2 * np.cos(s[0]+s[1]) ) \
                ]) / mg
            vg = (pgn - pgo) / dt
            vc = np.array([ns[6], ns[7]])
            ns[6:8] = ( (1.0+ec)*mg*vg + (mc-mg*ec)*vc ) / (mg + mc)
            # position update (arm position is assumed to be constant)
            if istip and not(isl2):
                ns[4:6] += ns[6:8]*dt
            else:
                # judge whether penetrate or not
                if (dp[0]*s[5] - dp[1]*s[4]) * (dp[0]*ns[5] - dp[1]*ns[4]) < 0.0:
                    pc = np.array([s[4], s[5]])
                    d2 = np.linalg.norm(pc-i2)
                dd = sz - d2
                ro = np.arctan2( (pc[1]-i2[1]) , (pc[0]-i2[0]) )
                ns[4] = pc[0] + dd * np.cos(ro) + ns[6]*dt
                ns[5] = pc[1] + dd * np.sin(ro) + ns[7]*dt
        elif isl1:
            # velocity update (arm velocity is assumed to be constant)
            ec = 1.0
            mg = m1
            pgn = np.array([lc1 * np.sin(ns[0]),  - lc1 * np.cos(ns[0])])
            pgo = np.array([lc1 * np.sin(s[0]),  - lc1 * np.cos(s[0])])
            vg = (pgn - pgo) / dt
            vc = np.array([ns[6], ns[7]])
            ns[6:8] = ( (1.0+ec)*mg*vg + (mc-mg*ec)*vc ) / (mg + mc)
            # position update (arm position is assumed to be constant)
            # judge whether penetrate or not
            if (p1[0]*s[5] - p1[1]*s[4]) * (p1[0]*ns[5] - p1[1]*ns[4]) < 0.0:
                pc = np.array([s[4], s[5]])
                d1 = np.linalg.norm(pc-i1)
            dd = sz - d1
            ro = np.arctan2( (pc[1]-i1[1]) , (pc[0]-i1[0]) )
            ns[4] = pc[0] + dd * np.cos(ro) + ns[6]*dt
            ns[5] = pc[1] + dd * np.sin(ro) + ns[7]*dt

        return ns

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
            self.viewer.set_bounds(-self.MAX_OBJ_POS,self.MAX_OBJ_POS,-self.MAX_OBJ_POS,self.MAX_OBJ_POS)

        p1 = [self.LINK_LENGTH_1 * np.sin(s[0]),
              -self.LINK_LENGTH_1 *np.cos(s[0])]

        p2 = [p1[0] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1]),
              p1[1] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1])]

        xys = [[0,0], p1, p2]
        thetas = [s[0]-0.5*np.pi, s[0]+s[1]-0.5*np.pi]
        links = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_circle(radius=self.LMAX, res=30, filled=False)
        for ((x,y),th,link) in zip(xys, thetas, links):
            l,r,t,b,c = 0, link, self.LINK_SIZE, -self.LINK_SIZE, self.LINK_SIZE
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0.0, 0.0, 0.8)
            circ = self.viewer.draw_circle(c)
            circ.set_color(0.0, 0.8, 0.0)
            circ.add_attr(jtransform)

        circ = self.viewer.draw_circle(0.5*self.OBJ_SIZE)
        circ.set_color(0.0, 0.0, 0.0)
        circ.add_attr(rendering.Transform(rotation=0.0, translation=(self.OBJ_REST[0], self.OBJ_REST[1])))
        circ = self.viewer.draw_circle(self.OBJ_SIZE)
        circ.set_color(0.8, 0.0, 0.0)
        circ.add_attr(rendering.Transform(rotation=0.0, translation=(s[4], s[5])))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
