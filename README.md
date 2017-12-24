# gym_multiobjective

# Dependency

[OpenAI Gym](https://github.com/openai/gym)

# Installation

```bash
git clone https://github.com/kbys-t/gym_MO.git
cd gym_MO
pip install -e .
```

# How to use
1. First of all,
`import gym_multiobjective`

1. Select environment from `["CartPoleMO-v0", "AcrobotMO-v0", "AcrobotMO-v1", "BallArmMO-v0", "BallArmMO-v1"]`
```python
ENV_NAME = "AcrobotMO-v0"
env = gym.make(ENV_NAME)
```

1. Prepare objectives
```python
task_name = env.TASK_NAME
objective = np.zeros(env.TASK_NUM)
objective[0] = 1.0  # choice from 0 ~ env.TASK_NUM-1
```
It's desired to normalize objective to make reward within [-1, 1]  

1. Send objectives together with action
```python
action = np.concatenate((action, objective))
observation, reward,, done, info = env.step(action)
```
If objectives are not sent, the same types of reward as OpenAI Gym will be returned basically.
