from time import time
import cv2
import sys

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer

def print_progress_bar(iteration, total, length=40):
    percent = (iteration / total)
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent:.2%}')
    sys.stdout.flush()

class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self.count_debug = 0

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			steps = 0
			while not done:
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				for i in range(0, obs.shape[0], 3):
					image = obs[i:i+3].numpy()
					flipped_image = cv2.flip(cv2.cvtColor(np.transpose(image, (1, 2, 0)), cv2.COLOR_BGR2RGB), 0)
					resized_image = cv2.resize(flipped_image, (512, 512), interpolation=cv2.INTER_AREA)
					cv2.imshow('Images', resized_image)
					cv2.waitKey(1)
				self.count_debug += 1
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
				steps += 1
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		), batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, True
		offline_steps = 50_000
		print(f"\nRunning Offline Training for {offline_steps} steps!\n")

		for step in range(offline_steps):
			_train_metrics = self.agent.update(self.buffer)
			print_progress_bar(step + 1, offline_steps)
			if step % (offline_steps // 10) == 0:
				train_metrics.update(_train_metrics)
	
		print(f"\nOffline training done, now online finetuning!")
		
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				for i in range(0, obs.shape[0], 3):
					image = obs[i:i+3].numpy()
					flipped_image = cv2.flip(cv2.cvtColor(np.transpose(image, (1, 2, 0)), cv2.COLOR_BGR2RGB), 0)
					resized_image = cv2.resize(flipped_image, (512, 512), interpolation=cv2.INTER_AREA)
					cv2.imshow('Images', resized_image)
					cv2.waitKey(1)
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			for i in range(0, obs.shape[0], 3):
				image = obs[i:i+3].numpy()
				flipped_image = cv2.flip(cv2.cvtColor(np.transpose(image, (1, 2, 0)), cv2.COLOR_BGR2RGB), 0)
				resized_image = cv2.resize(flipped_image, (512, 512), interpolation=cv2.INTER_AREA)
				cv2.imshow('Images', resized_image)
				cv2.waitKey(1)
			self._tds.append(self.to_td(obs, action, reward))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print(f'Pretraining agent on seed data for {num_updates} steps')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1
	
		self.logger.finish(self.agent)
