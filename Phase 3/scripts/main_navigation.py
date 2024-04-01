import isaacgym
assert isaacgym
import torch

# from go1_gym.envs.base.legged_robot_config import Cfg
# from go1_gym.env.base.navigation_robot_config import Cfg
# from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.position_tracking import PositionTrackingEasyEnv

from ml_logger import logger

from go1_gym_learn.ppo_cse_nav import Runner
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go1_gym_learn.ppo_cse_nav.actor_critic import AC_Args
from go1_gym_learn.ppo_cse_nav.ppo import PPO_Args
from go1_gym_learn.ppo_cse_nav import RunnerArgs

global final_path
# def nav_config_go1(Cfg):
#   # make necessary changes
#   # used to update the x_init_ , y_init , yaw changes

def train_go1(headless=True):
  from go1_gym.envs.navigation.navigation_robot_config import Cfg
  # nav_config_go1(Cfg) # TODO # to make changes

  # TODO : config setting from train.py should be verified

  env = PositionTrackingEasyEnv(sim_device='cuda:0', headless=True, cfg=Cfg)

  # log the experiment parameters
  logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                    Cfg=vars(Cfg))

  env = HistoryWrapper(env)
  gpu_id = 0
  runner = Runner(env, device=f"cuda:{gpu_id}")
  num_learning_iterations = 50
  runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True, eval_freq=100)
  
  # print("THE AVERAGE REWARDS THAT IS SEEN IS " , runner.reward_plot_tracker_per_iteration)
  training_rewards_per_iteration = runner.reward_plot_tracker_per_iteration
  success_rate_per_iteration = runner.success_plot_tracker_per_iteration
  num_rewards = len(training_rewards_per_iteration.keys())

  from matplotlib import pyplot as plt
  import numpy as np
  plt.rcParams.update({'font.size': 7})
  fig, axs = plt.subplots(num_rewards, 1, figsize=(12, 15))
  line_space_intervals = num_learning_iterations if num_learning_iterations<= 1000 else 1000 

  for i,x in enumerate(training_rewards_per_iteration.keys()):
      axs[i].plot(np.linspace(0, num_learning_iterations, line_space_intervals), training_rewards_per_iteration[x], color='black', linestyle="-", label="Reward")
      # axs[0].legend()
      axs[i].set_title(x)
      axs[i].set_xlabel("Iterations")
      axs[i].set_ylabel("Direct Reward values")
  
  fig.savefig(f"{final_path}/training_rewards.png")
  fig.savefig(f"{final_path}/training_rewards.jpg")
  for ax in axs:
      ax.grid(True)   
  plt.subplots_adjust( hspace=0.5)
  plt.tight_layout()
  plt.show()

  print("success_rate_per_iteration", success_rate_per_iteration)
  
  from matplotlib import pyplot as plt
  plt.plot(np.linspace(0, num_learning_iterations, line_space_intervals),success_rate_per_iteration )
  plt.xlabel('Iterations')
  plt.ylabel('Success Rates')
  plt.title('Success per Iterations')
  # Save the plot as an image file (e.g., PNG, JPEG, PDF)
  plt.savefig(f"{final_path}/succes_per_iteration.png")  # Change the file extension as needed (e.g., .jpg, .pdf)
  plt.savefig(f"{final_path}/succes_per_iteration.jpg")
  plt.show()


if __name__ == '__main__':
    from pathlib import Path
    from ml_logger import logger
    from go1_gym import MINI_GYM_ROOT_DIR
    from datetime import datetime

    stem = Path(__file__).stem
    current_datetime = datetime.utcnow()
    log_file_path = f"navigation-policy/{current_datetime.strftime('%Y-%m-%d')}/{stem}/{current_datetime.strftime('%H%M%S.%f')}"
    logger.configure(logger.utcnow(log_file_path),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    final_path = f"{MINI_GYM_ROOT_DIR}/runs/{log_file_path}"
    logger.log_text("""
                charts: 
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_lin_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_1/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_2/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_orientation_control/mean
                  xKey: iterations
                - yKey: train/episode/rew_dof_pos/mean
                  xKey: iterations
                - yKey: train/episode/command_area_trot/mean
                  xKey: iterations
                - yKey: train/episode/max_terrain_height/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                - yKey: adaptation_loss/mean
                  xKey: iterations
                """, filename=".charts.yml", dedent=True)

    # to see the environment rendering, set headless=False
    train_go1(headless=True)
