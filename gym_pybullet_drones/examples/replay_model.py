import os
import time
from datetime import datetime
import argparse
import re
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.envs.GroupAviary import GroupAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger

from gym_pybullet_drones.utils.enums import ActionType, ObservationType

if __name__ == "__main__":

    folder = r'C:\Users\nils\OneDrive - Duke University\Documents\Robot Learning\gym-pybullet-drones\results\save-12.19.2023_03.52.38'

    #### Load the model from file ##############################

    if os.path.isfile(folder+'/success_model.zip'):
        path = folder+'/success_model.zip'
    elif os.path.isfile(folder+'/best_model.zip'):
        path = folder+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", folder)
    model = PPO.load(path)


    #### Parameters to recreate the environment ################
    OBS = ObservationType.KIN
    ACT = ActionType.PID
    AGENTS = 3

    #### Evaluate the model ####################################
    eval_env = make_vec_env(GroupAviary,
                                env_kwargs=dict(num_drones=AGENTS, obs=OBS, act=ACT),
                                n_envs=1,
                                seed=0
                                )
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    #### Show, record a video, and log the model's performance #
    test_env = GroupAviary(gui=True,
                            num_drones=AGENTS,
                            obs=OBS,
                            act=ACT)
    test_env_nogui = GroupAviary(num_drones=AGENTS, obs=OBS, act=ACT)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=AGENTS,
                )
    mean_reward, std_reward = evaluate_policy(model,
                                              test_env,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if OBS == ObservationType.KIN:
            for d in range(AGENTS):
                logger.log(drone=d,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=np.hstack([obs2[d][0:3],
                                        np.zeros(4),
                                        obs2[d][3:15],
                                        act2[d]
                                        ]),
                    control=np.zeros(12)
                    )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    #### Print training progression ############################
    with np.load(folder+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    if OBS == ObservationType.KIN:
        logger.plot()

    # with np.load(ARGS.exp+'/evaluations.npz') as data:
    #     print(data.files)
    #     print(data['timesteps'])
    #     print(data['results'])
    #     print(data['ep_lengths'])
