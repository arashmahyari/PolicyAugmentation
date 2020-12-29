# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 22:57:35 2020

@author: arash
"""

import gym, sys
import matplotlib.pyplot as plt
#sys.path.append('E:/RL/NEW SEP 2020/New Updated Master')
import pickle
#from stable_baselines3 import DQN
from stable_baselines import PPO1, PPO2
from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines.common import make_vec_env
total_rewards=[]

for loop in range(10):
    print('***** LOOP ', loop)
    # Create environment
    env = gym.make('MountainCar-v0')
    #env = make_vec_env('MountainCar-v0', n_envs=4)
    
    # Instantiate the agent
    model = PPO2('MlpPolicy', env, verbose=0, n_steps=200)
    #model.learning_starts=10
    #model.target_update_interval=200
    #model.batch_size=64
    
    # Train the agent
    #model.learn(total_timesteps=int(2e5))
    model.learn(total_timesteps=int(40000))
    
    
    
    # print('******************************')
    # print('******************************')
    # test_reward_total=[]
    # for i in range(10):
    #     obs = env.reset()
    #     test_reward=0
    #     for j in range(200):
                
    #         action, _states = model.predict(obs)
    #         obs, rewards, dones, info = env.step(action)
    #         if obs[0]>0.48:
    #             rewards=10
    #             print('****** reward ********')
    #         test_reward+=rewards    
    #         env.render()
    #     test_reward_total.append(test_reward)    
    
    total_rewards.append(model.episode_rewards)
    #pickle.dump([total_rewards],open('PPO_Mountain_ICM3.p','wb'))

#plt.plot(model.training_rewrad)

# t=list(range(100))
# total_rewards=np.array(total_rewards)[:,:100]
# fig, ax = plt.subplots(1)
# ax.plot(t,np.mean(total_rewards,axis=0), lw=2, label='mean population 1', color='blue')
# ax.fill_between(t,np.mean(total_rewards,axis=0)+np.std(total_rewards,axis=0), np.mean(total_rewards,axis=0)-np.std(total_rewards,axis=0), facecolor='yellow', alpha=0.5)




# Save the agent
#model.save("dqn_lunar")
#del model  # delete trained model to demonstrate loading

# Load the trained agent
#model = DQN.load("dqn_lunar")



# Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# # Enjoy trained agent
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     print(action)
#     env.render()