#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:58:21 2024

@author: joetao
"""
import numpy as np
import sys
import math
import torch
import torch.nn.functional as F
import torch.optim as optim

from Env import Env_multi_sim_img
from model import VesNet_DQN
from collections import deque
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import random
print(torch.__version__)
LR = 5e-4
max_step=500
n_episodes=300
gamma=0.99
gae_lambda=1.0
entropy_coef=0.01
value_loss_coef=0.5
max_grad_norm=50
save_every=50
update_every=20
TAU = 0.0025

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def create_configs_rand(num):
    configs=[]
    r_min=30
    r_max=75
    for i in range(num):
        offset=np.random.rand()*np.pi/2
        size_3d=[750,700,450]
        r=np.random.randint(r_min+(r_max-r_min)*i/num,r_min+(r_max-r_min)*(i+1)/num)
        c_x=350
        c_y=np.random.randint(50+r,225)
        c=[c_x,c_y]
        config=(c,r,size_3d,offset)
        configs.append(config)
    return configs

configs=create_configs_rand(10)
print(configs)
env=Env_multi_sim_img(configs=configs, num_channels=4)
print('env created')
policy_net = VesNet_DQN(env.num_channels, 5, env.num_actions).to(device)
target_net = VesNet_DQN(env.num_channels, 5, env.num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.SGD(policy_net.parameters(), lr=LR)

scores_window = deque(maxlen=50)
rewards_window = deque(maxlen=50)
rewards_his=[]
smoothend_rewards=[]

a_file = open("VesNet_RL_ckpt/configs.txt", "w")
for row in configs:
    a_file.write(str(row)+'\r')
a_file.close()


plt.clf()
plt.ylabel('Score')
plt.xlabel('Episode #')
done_his=deque(maxlen=100)
reward_max=-sys.float_info.max
best_success_rate=0
i_episode=0
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000


criterion = nn.SmoothL1Loss()

print("start training")
for i_episode in tqdm(range(n_episodes)):
    print("episode:", i_episode)
    if i_episode==1500:
        for g in optimizer.param_groups:
            g['lr'] = 1e-4
    elif i_episode==500:
        for g in optimizer.param_groups:
            g['lr'] = 3e-4
    elif i_episode==0:
        for g in optimizer.param_groups:
            g['lr'] = 5e-4
    
    cx = torch.zeros(1, 256).float().to(device)
    hx = torch.zeros(1, 256).float().to(device)
    state = env.reset(randomVessel=False)
    reward_sum=0
    finish=False
    done = False
    t=0

    values = []
    log_probs = []
    rewards = []
    entropies = []
    while not done:
        # for _ in range(0,update_every):
        t+=1
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * t / EPS_DECAY)
        if random.random() < eps_threshold:
            with torch.no_grad():
        #     # t.max(1) will return the largest column value of each row.
        #     # second column on max result is index of where max element was
        #     # found, so we pick action with the larger expected reward.
                action = policy_net((state,(hx, cx)))[0].max(1).indices.view(1, 1)
        else:
            action = torch.tensor([[random.randint(0, 6)]], device=device, dtype=torch.long)
        # print(action)
        Q_st, (hx, cx) = policy_net((state,(hx.detach(), cx.detach())))
        Q_st_a = Q_st.gather(1, action)
        # print(Q_st_a)
        state, reward, finish_ = env.step(int(action.cpu().detach().numpy()))
        # print(reward)
        next_state_value = target_net((state,(hx.detach(), cx.detach())))[0].max(1)[0].detach()
        finish=finish or finish_
        done = (t >= max_step) or finish
        # print(t,done,finish)
        Q = (next_state_value * gamma) + reward
        # print(Q)
        loss = criterion(Q_st_a, Q.reshape(-1,1))

        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

        rewards.append(reward)
        
        reward_sum+=reward

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            break

    done_his.append(finish)
    scores_window.append(reward_sum)
    rewards_window.append(reward_sum)
    rewards_his.append(reward_sum)
    smoothend_rewards.append(np.mean(scores_window))
    
    if np.mean(rewards_window)>reward_max:
        reward_max=np.mean(rewards_window)
        torch.save(target_net.state_dict(), 'VesNet_RL_ckpt/checkpoint.pth')
        a_file = open("VesNet_RL_ckpt/best_ckpt.txt", "w")
        a_file.write(str([i_episode,reward_max])+'\r')
        a_file.close()
    
    if i_episode%save_every==0:
        torch.save(target_net.state_dict(), 'VesNet_RL_ckpt/checkpoint_latest.pth')
    success_rate=len(np.where(np.array(done_his)==1)[0])
    
    if success_rate>best_success_rate:
        best_success_rate=success_rate
        if i_episode>100:
            torch.save(target_net.state_dict(), 'VesNet_RL_ckpt/checkpoint_best_sr.pth')
            a_file = open("VesNet_RL_ckpt/best_ckpt_sr.txt", "w")
            a_file.write(str([i_episode,best_success_rate])+'\r')
            a_file.close()
    print('Episode %d Average Score: %.2f Score: %.2f Steps: %d Done: %s Best success rate: %d %% Success rate: %d %%\r' % 
                            (i_episode,np.mean(scores_window),reward_sum,t,finish,best_success_rate,success_rate))
    
    
    if len(rewards_his)>2:
        plt.plot([len(rewards_his)-1,len(rewards_his)], [rewards_his[-2],rewards_his[-1]],'b', alpha=0.3)
        plt.plot([len(smoothend_rewards)-1,len(smoothend_rewards)], [smoothend_rewards[-2],smoothend_rewards[-1]],'g')
        plt.pause(0.05)
    
    t = 0
    if i_episode>50:
        state = env.reset(randomVessel=True)
    else:
        state = env.reset(randomVessel=True)
    i_episode+=1
    reward_sum=0
    finish=False
        

torch.save(target_net.state_dict(), 'VesNet_RL_ckpt/checkpoint_latest.pth')
plt.savefig('VesNet_RL_ckpt/plot.png')