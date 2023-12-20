# Swarm-Learning

This repository is based on the code at https://github.com/utiasDSL/gym-pybullet-drones/tree/master and as defined in the following paper: https://arxiv.org/abs/2103.02142

### Manual Control

The swarm_motion_ctrl.py and swarm_motion_sim.py scripts in gym_pybullet_drones/examples allow for manual control of an arbitrary number of drones (currently setup for 9 drones). Run this manual control simuilation by navigating to the scripts' directory, then view motion in simulation by running with

```sh
python swarm_motion_sim.py
```

### Swarm Learning

In order to train a model, use the gym_pybullet_drones/examples/grouplearn.py script. This script (based on the newly implemented GroupAviary) will train a model, test it, and create a zip file with the testing results in a results directory initialized at the current working directory. To run, navigate to the gym_pybullet_drones/examples directory and use

```sh
python grouplearn.py
```
If the script is run from another directory, a results subdirectory will be created and the results placed in there under the running time of the script.

To adjust time spent training, modify the total_timesteps parameter in the call to model.learn. Training with 3,000,000 timesteps currently results in a training time of a few hours. For a quick training model, use a timestep amount in the region of 30,000 or slightly more.

To view a "replay" of the model's performance by conducting a test run, use the script replay_model.py in the same directory with 

```sh
python replay_model.py
```
By modifying the filepath of variable "folder" to the desired subdirectory in the results folder, the model from that run will be used.
