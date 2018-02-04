import logging
from gym.envs.registration import registry, register, make, spec

logger = logging.getLogger(__name__)

register(
id='PendulumMO-v0',
entry_point='gym_multiobjective.envs:PendulumBalanceEnv',
)

register(
id='PendulumMO-v1',
entry_point='gym_multiobjective.envs:PendulumSwingEnv',
)

register(
    id='CartPoleMO-v0',
    entry_point='gym_multiobjective.envs:CartPoleBalanceEnv',
)

register(
    id='CartPoleMO-v1',
    entry_point='gym_multiobjective.envs:CartPoleSwingEnv',
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
