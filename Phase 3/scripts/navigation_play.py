import isaacgym

assert isaacgym
import torch
import numpy as np

import math
from isaacgym.torch_utils import *



import glob
import pickle as pkl

from go1_gym.envs import *

from go1_gym.envs.navigation.navigation_robot_config import Cfg
from go1_gym.envs.go1.position_tracking import PositionTrackingEasyEnv

from tqdm import tqdm
from go1_gym.envs.rewards.corl_rewards import quat_vector_cosine

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action
    
    return policy    


def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[-1]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)
    
    Cfg.env.num_envs = 1

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = PositionTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    label = "navigation-policy/2023-12-04/main_navigation"
    num_eval_steps = 750

    # # reward function emulators
    # def _reward_tracking_lin_pos():
    #     lin_pos_error = torch.sum(torch.square(env.commands[:, :1] - env.base_pos[:, :1]), dim=1)
    #     return torch.exp(-lin_pos_error / env.cfg.rewards.tracking_sigma)[0].cpu().numpy()

    # def _reward_goal_body_alignment():
    #     goal_direction = env.commands[:, :2] - env.base_pos[:, :2]
    #     goal_direction = torch.nn.functional.pad(goal_direction, (0, 1), mode='constant', value=0)
    #     return quat_vector_cosine(env.base_quat, goal_direction)[0].cpu().numpy()

    # # additional reward for magnitude of base_lin_vel in the direction of goal
    # def _reward_direct_navigation():
    #     goal_direction = env.commands[:, :2] - env.base_pos[:, :2]
    #     goal_direction = torch.nn.functional.pad(goal_direction, (0, 1), mode='constant', value=0)
    #     cosine = quat_vector_cosine(env.base_quat, goal_direction)

    #     velocity_magnitude = torch.sum(torch.square(env.locomotion_env.base_lin_vel[:, :2]), dim=1)
    #     return (cosine * velocity_magnitude)[0].cpu().numpy()

    # def _reward_reach_goal():
    #     # Reward robot crossing goal line
    #     return 1.0 if env.base_pos[0][0].cpu().numpy() > 3.5 else 0

    # tracking_lin_pos_reward = 0
    # goal_body_alignment_reward = 0
    # reach_goal_reward = 0
    # direct_navigation_reward = 0

    # rewards = {
    #     'tracking_lin_pos': [],
    #     'tracking_lin_pos_direct': [],
    #     'reach_goal': [],
    #     'goal_body_alignment': [],
    #     'direct_navigation': [],
    #     'total': []
    # }
    
    # x_vel_cmd_array = np.zeros(num_eval_steps)
    # y_vel_cmd_array = np.zeros(num_eval_steps)
    # for i in range(num_eval_steps):
    #     x_vel_cmd_array[i] = 3

    env, policy = load_env(label, headless=headless)
    
    obs = env.reset()
    print(obs)
    frames = []

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        # env.commands[:, 0] = 3
        # env.commands[:, 1] = 0

        obs, rew, done, info = env.step(actions)

        frames.append(env.render())

        # tracking_lin_pos_reward += _reward_tracking_lin_pos() * 0.8
        # reach_goal_reward += _reward_reach_goal() * 20.0
        # goal_body_alignment_reward += _reward_goal_body_alignment() * 0.02
        # direct_navigation_reward = _reward_direct_navigation() * 0.01
        # rewards['tracking_lin_pos_direct'].append(_reward_tracking_lin_pos() * 0.8)
        # rewards['tracking_lin_pos'].append(tracking_lin_pos_reward)
        # rewards['reach_goal'].append(reach_goal_reward)
        # rewards['goal_body_alignment'].append(goal_body_alignment_reward)
        # rewards['direct_navigation'].append(direct_navigation_reward)
        # rewards['total'].append(rewards['tracking_lin_pos'][-1] + rewards['reach_goal'][-1] + rewards['goal_body_alignment'][-1] + rewards['direct_navigation'][-1])

    if len(frames) > 0:
        print("LOGGING VIDEO")
        logger.save_video(frames, "nav_play_demo.mp4", fps=1 / env.dt)

    # print("I AM PRINTING THE REWARDS ACCUMULATED OVER TIME\n ", self.reward_plot_tracker )

    # num_eval_steps = len(rewards['tracking_lin_pos']) #for cases it breaks out of the loop early
    # plot target and measured forward velocity
    # from matplotlib import pyplot as plt
    # plt.rcParams.update({'font.size': 7})
    # fig, axs = plt.subplots(6, 1, figsize=(12, 15))

    # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), rewards['tracking_lin_pos_direct'], color='black', linestyle="-", label="Reward")
    # # axs[0].legend()
    # axs[0].set_title("tracking_lin_pos")
    # axs[0].set_xlabel("Iterations")
    # axs[0].set_ylabel("Direct Reward values")

    # axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), rewards['tracking_lin_pos'], color='red', linestyle="-", label="Reward")
    # # axs[1].legend()
    # axs[1].set_title("Reward tracking_lin_pos")
    # axs[1].set_xlabel("Iterations")
    # axs[1].set_ylabel("Cumulative Reward")

    # axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), rewards['goal_body_alignment'], color='red', linestyle="-", label="Reward")
    # axs[2].set_title("Reward goal_body_alignment")
    # axs[2].set_xlabel("Iterations")
    # axs[2].set_ylabel("Cumulative Reward")

    # axs[3].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), rewards['direct_navigation'], color='black', linestyle="-", label="Reward")
    # axs[3].set_title("Reward direct_navigation")
    # axs[3].set_xlabel("Iterations")
    # axs[3].set_ylabel("Direct Reward values")

    # axs[4].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), rewards['reach_goal'], color='red', linestyle="-", label="Reward")
    # axs[4].set_title("Reward reach_goal")
    # axs[4].set_xlabel("Iterations")
    # axs[4].set_ylabel("Cumulative Reward")
    
    # axs[5].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), rewards['total'], color='red', linestyle="-", label="Reward")
    # axs[5].set_title("Reward total")
    # axs[5].set_xlabel("Iterations")
    # axs[5].set_ylabel("Cumulative Reward")

    # for ax in axs:
    #     ax.grid(True)   
    # plt.subplots_adjust( hspace=0.5)
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)