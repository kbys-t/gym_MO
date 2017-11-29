import logging
from gym.envs.registration import registry, register, make, spec

logger = logging.getLogger(__name__)

register(
    id='CartPoleMO-v0',
    entry_point='gym_multiobjective.envs:CartPoleEnv',
)

register(
    id='AcrobotMO-v0',
    entry_point='gym_multiobjective.envs:AcrobotBalanceEnv',
)

register(
    id='AcrobotMO-v1',
    entry_point='gym_multiobjective.envs:AcrobotSwingEnv',
)

register(
    id='BallArmMO-v0',
    entry_point='gym_multiobjective.envs:BallArmStaticEnv',
)

register(
    id='BallArmMO-v1',
    entry_point='gym_multiobjective.envs:BallArmDynamicEnv',
)
