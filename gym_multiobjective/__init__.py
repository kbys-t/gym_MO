import logging
from gym.envs.registration import registry, register, make, spec

logger = logging.getLogger(__name__)

register(
    id='AcrobotMO-v0',
    entry_point='gym_multiobjective.envs:AcrobotEnv',
)

register(
    id='AcrobotMO-v1',
    entry_point='gym_multiobjective.envs:AcrobotSWEnv',
)

register(
    id='CartPoleMO-v0',
    entry_point='gym_multiobjective.envs:CartPoleEnv',
)
