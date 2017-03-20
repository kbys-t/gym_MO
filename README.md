# gym_multiobjective


# Installation

```bash
cd gym_multiobjective
pip install -e .
```

# How to use
1. First of all,
`import gym_multiobjective`

1. Prepare objectives
```python
objective = np.array([0.1, 1.0, 0.0])
objective = objective / np.linalg.norm(objective, 1)
```
It's desired to normalize objective to make reward within [-1, 1]  
Here,
  + 0-th is for motion minimization,
  + 1-th is for height,
  + 2-th is for angular velocity

1. Send objectives together with action
```python
action = np.concatenate((action, objective))
observation, reward,, done, info = env.step(action)
```
If objectives are not sent, the same types of reward as OpenAI Gym will be returned.
