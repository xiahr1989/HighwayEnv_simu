# HighwayEnv

## System
Windows, Python 3.12

## Execution
run scripts\sb3_highway_dqn.py, the DQN model and the videos will be saved in folder scripts\highway_dqn\

## Changes I made are listed as follows:
### envs
1) highway-env.py: modified for integration
2) action.py: modified for the convenience of rule-design
### rule_hierarchy
1) demo.py: modified to customize the rules in the form of STL
2) rule_hierarchy.py: traj_cost is modified to calculate reward for each rule

core parts are in highway-env.py and demo.py