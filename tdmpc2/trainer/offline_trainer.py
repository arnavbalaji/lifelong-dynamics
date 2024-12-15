import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob
import threading

import numpy as np
import torch
from tqdm import tqdm

from common.buffer import Buffer
from trainer.base import Trainer
import torchvision
import h5py
from tensordict.tensordict import TensorDict
import cv2
from collections import deque

lock = threading.Lock()


class OfflineTrainer(Trainer):
	"""Trainer class for multi-task offline TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
		os.makedirs(f"/home/arpit/projects/tdmpc2/{self.cfg.eval_vid_directory}")
	
	# def to_td(self, obs, action, reward, task=None):
	# 	"""Creates a TensorDict for a new episode."""
	# 	if isinstance(obs, dict):
	# 		obs = TensorDict(obs, batch_size=(), device='cpu')
	# 	else:
	# 		obs = obs.unsqueeze(0).cpu()
	# 	if task is not None:
	# 		td = TensorDict(dict(
	# 			obs=obs,
	# 			action=action.unsqueeze(0),
	# 			reward=reward.unsqueeze(0),
	# 			task=torch.tensor(task).unsqueeze(0)
	# 		), batch_size=(1,))
	# 		return td
	# 	td = TensorDict(dict(
	# 		obs=obs,
	# 		action=action.unsqueeze(0),
	# 		reward=reward.unsqueeze(0),
	# 	), batch_size=(1,))
	# 	return td
	def to_td(self, obs, action=None, reward=None, dummy_action=None):
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
	
	def eval(self, train_index=500_000):
		"""Evaluate a TD-MPC2 agent."""
		results = dict()
		images = []
		for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
			ep_rewards, ep_successes = [], []
			for ep_no in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
				while not done:
					# action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
					action = self.agent.act(obs, t0=t==0, eval_mode=True)
					obs, reward, done, info = self.env.step(action)
					for i in range(0, obs.shape[0], 3):
						image = obs[i:i+3].numpy()
						flipped_image = cv2.flip(cv2.cvtColor(np.transpose(image, (1, 2, 0)), cv2.COLOR_BGR2RGB), 0)
						resized_image = cv2.resize(flipped_image, (512, 512), interpolation=cv2.INTER_AREA)
						save_image = cv2.resize(flipped_image, (128, 128), interpolation=cv2.INTER_AREA)
						if train_index % 100_000 == 0 and ep_no < 2:
							with lock:
								images.append(save_image)
						cv2.imshow('Images', resized_image)
						cv2.waitKey(1)
					ep_reward += reward
					t += 1
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
			results.update({
				f'episode_reward+{self.cfg.tasks[task_idx]}': np.nanmean(ep_rewards),
				f'episode_success+{self.cfg.tasks[task_idx]}': np.nanmean(ep_successes),})
		return results, images
				
	def train(self):
		"""Train a TD-MPC2 agent."""
		assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80', 'libero_task'}, \
			'Offline training only supports multitask training with mt30 or mt80 task sets.'

		# Load data
		if self.cfg.task == "libero_task":
			_cfg = deepcopy(self.cfg)
			_cfg.episode_length = 501
			_cfg.steps = _cfg.buffer_size
			self.buffer = Buffer(_cfg)

			transform = torchvision.transforms.Resize((64, 64))

			hdf5_directory_name = f"/home/arpit/projects/tdmpc2/LIBERO/libero/datasets/{_cfg.task_suite}"

			for task_id in range(len(self.env.envs)):
				temp_env = self.env.envs[task_id]
				task_name = temp_env.task.name
				# filename = f"{hdf5_directory_name}/{task_name}_extra_data.hdf5"
				filename = f"{hdf5_directory_name}/{task_name}_extra_data_only_success2.hdf5"
				transform = torchvision.transforms.Resize((64, 64))
				with h5py.File(filename) as file:
					print(f"Adding success data from {filename}")
					data = file["data"]
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
						tds = [self.to_td(obs, dummy_action=demo_actions[0].to(torch.float32))]

						for i in range(demo_actions.shape[0]):
							frames.append(demo_images[i+1])
							obs = torch.from_numpy(np.concatenate(frames))
							tds.append(self.to_td(obs, demo_actions[i].to(torch.float32), (demo_rewards[i] - 1.).to(torch.float32)))

						self.buffer.add(torch.cat(tds))
		else:
			assert self.cfg.task in self.cfg.data_dir, \
				f'Expected data directory {self.cfg.data_dir} to contain {self.cfg.task}, ' \
				f'please double-check your config.'
			fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
			fps = sorted(glob(str(fp)))
			assert len(fps) > 0, f'No data found at {fp}'
			print(f'Found {len(fps)} files in {fp}')
		
			# Create buffer for sampling
			_cfg = deepcopy(self.cfg)
			_cfg.episode_length = 101 if self.cfg.task == 'mt80' else 501
			_cfg.buffer_size = 550_450_000 if self.cfg.task == 'mt80' else 345_690_000
			_cfg.steps = _cfg.buffer_size
			self.buffer = Buffer(_cfg)
			for fp in tqdm(fps, desc='Loading data'):
				td = torch.load(fp)
				assert td.shape[1] == _cfg.episode_length, \
					f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
					f'please double-check your config.'
				for i in range(len(td)):
					self.buffer.add(td[i])
			assert self.buffer.num_eps == self.buffer.capacity, \
				f'Buffer has {self.buffer.num_eps} episodes, expected {self.buffer.capacity} episodes.'
		
		print(f'Training agent for {self.cfg.steps} iterations...')
		metrics = {}
		for i in range(self.cfg.steps):

			# Update agent
			train_metrics = self.agent.update(self.buffer)

			# Evaluate agent periodically
			if i % self.cfg.eval_freq == 0 or i % 10_000 == 0:
				metrics = {
					'iteration': i,
					'total_time': time() - self._start_time,
				}
				metrics.update(train_metrics)
				if i % self.cfg.eval_freq == 0:
					eval_results, images = self.eval(i)
					metrics.update(eval_results)
					with lock:
						if len(images) > 0:
							height, width, _ = images[0].shape
							size = (width, height)
							fps = 60 
							output_filename = f"/home/arpit/projects/tdmpc2/{self.cfg.eval_vid_directory}/{i}.avi"
							out = cv2.VideoWriter(output_filename, 0, fps, size)
							for img in images:
								out.write(img)
							out.release()
							cv2.destroyAllWindows()
						images.clear() # Trigger garbage collection
					self.logger.pprint_multitask(metrics, self.cfg)
					if i > 0:
						self.logger.save_agent(self.agent, identifier=f'{i}')
				self.logger.log(metrics, 'pretrain')
			
		self.logger.finish(self.agent)
