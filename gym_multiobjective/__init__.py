import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='AcrobotMO-v0',
    entry_point='gym_multiobjective.envs:AcrobotEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='CartPoleMO-v0',
    entry_point='gym_multiobjective.envs:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 500},
)
