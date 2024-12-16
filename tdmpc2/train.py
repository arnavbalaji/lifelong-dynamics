import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import h5py
import sys

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger
from tensordict.tensordict import TensorDict
from collections import deque

torch.backends.cudnn.benchmark = True

def print_progress_bar(iteration, total, length=40):
	'''
	Prints progress bar. Used when loading in demos to buffer. 
	'''
	percent = (iteration / total)
	filled_length = int(length * percent)
	bar = '█' * filled_length + '-' * (length - filled_length)
	sys.stdout.write(f'\r|{bar}| {percent:.2%}')
	sys.stdout.flush()

def to_td(obs, action=None, reward=None, dummy_action=None):
	"""Creates a TensorDict for a new episode."""
	if isinstance(obs, dict):
		obs = TensorDict(obs, batch_size=(), device='cpu')
	else:
		obs = obs.unsqueeze(0).cpu()
	if action is None:
		action = torch.full_like(dummy_action, float('nan')).to(torch.float32)
	if reward is None:
		reward = torch.tensor(float('nan')).to(torch.float32)
	td = TensorDict(dict(
		obs=obs,
		action=action.unsqueeze(0),
		reward=reward.unsqueeze(0),
	), batch_size=(1,))
	return td


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
	
	cfg.multitask = False # Running offline-online training
	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	env = make_env(cfg)
	train_buffer = Buffer(cfg)

	if cfg.task == "libero_task" and not cfg.multitask:
		hdf5_directory_name = f"{cfg.path_to_dir}/LIBERO/libero/datasets/{cfg.task_suite}"
		task_name = env.task.name

		filename = f"{hdf5_directory_name}/{task_name}_extra_data_200.hdf5"

		print(f"\nLoading demos from {filename} into buffer!\n")
		transform = torchvision.transforms.Resize((64, 64))
		with h5py.File(filename) as file:
			# Loading demonstration into buffer
			data = file["data"]
			num_demos = len(list(data.items()))
			iteration = 1
			for d in data.items():
				demo = d[1]
				demo_images = np.array(demo["obs"]["agentview_rgb"])
				demo_images = transform(torch.permute(torch.from_numpy(demo_images), (0, 3, 1, 2)))

				demo_actions = torch.from_numpy(np.array(demo["actions"]))
				demo_rewards = torch.from_numpy(np.array(demo["rewards"]))

				frames = deque([], maxlen=3)
				for _ in range(3):
					frames.append(demo_images[0])

				obs = torch.from_numpy(np.concatenate(frames))
				tds = [to_td(obs, dummy_action=demo_actions[0].to(torch.float32))]

				for i in range(demo_actions.shape[0]):
					frames.append(demo_images[i+1])
					obs = torch.from_numpy(np.concatenate(frames))
					tds.append(to_td(obs, demo_actions[i].to(torch.float32), (demo_rewards[i] - 1.).to(torch.float32)))

				train_buffer.add(torch.cat(tds))
				print_progress_bar(iteration, num_demos)
				iteration += 1

	trainer = trainer_cls(
		cfg=cfg,
		env=env,
		agent=TDMPC2(cfg),
		buffer=train_buffer,
		logger=Logger(cfg),
		show_images=cfg.show_images
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
