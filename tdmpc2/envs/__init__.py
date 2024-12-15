from copy import deepcopy
import warnings

import gym
import os

from envs.wrappers.multitask import MultitaskWrapper, LIBEROMultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from envs.dmcontrol import make_env as make_dm_control_env
except:
	make_dm_control_env = missing_dependencies
try:
	from envs.maniskill import make_env as make_maniskill_env
except:
	make_maniskill_env = missing_dependencies
try:
	from envs.metaworld import make_env as make_metaworld_env
except:
	make_metaworld_env = missing_dependencies
try:
	from envs.myosuite import make_env as make_myosuite_env
except:
	make_myosuite_env = missing_dependencies
try:
	from envs.libero_env import make_env as make_libero_env
except: 
	make_libero_env = missing_dependencies


warnings.filterwarnings('ignore', category=DeprecationWarning)

task_suites = ["libero_object", "libero_spatial", "libero_goal", "libero_10", "libero_90", "libero_100"]


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise ValueError('Unknown task:', task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env

def make_libero_multitask_env(cfg):
	"""
	Make a multi-task LIBERO environment for TD-MPC2 experiments.
	"""
	assert cfg.task_suite in task_suites, (f"{cfg.task_suite} not a valid task. Options are libero_object," 
                                " libero_spatial, libero_goal, libero_10, libero_100")
	print(f"Creating multitask LIBERO environment with tasks from {cfg.task_suite}")
	hdf5_directory_name = f"/home/arpit/projects/tdmpc2/LIBERO/libero/datasets/{cfg.task_suite}"
	envs = []
	tasks = []
	num_tasks = len([filename for filename in os.listdir(hdf5_directory_name) if filename.endswith(".hdf5")])
	for task_idx in range(1):
		_cfg = deepcopy(cfg)
		_cfg.task_id = task_idx
		_cfg.multitask = False
		env = make_libero_env(_cfg)
		if env is None:
			raise ValueError('Unknown task index:', task_idx)
		envs.append(env)
		tasks.append(env.task.name)
	cfg.tasks = tasks
	env = LIBEROMultitaskWrapper(cfg, envs)
	cfg.episode_lengths = env._episode_lengths
	cfg.action_dims = env._action_dims
	return env
	

def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gym.logger.set_level(40)
	print(f"\ncfg.multitask = {cfg.multitask}\n")
	if cfg.multitask:
		# env = make_multitask_env(cfg)
		print("\nYO\n")
		env = make_libero_multitask_env(cfg)

	else:
		env = None
		if cfg.task == "libero_task":
			try:
				env = make_libero_env(cfg)
			except:
				pass
			# env = make_libero_env(cfg)
		else:
			for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env]:
				try:
					env = fn(cfg)
				except ValueError:
					pass
		if env is None:
			raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
	env = TensorWrapper(env)
	if cfg.get('obs', 'state') == 'rgb':
		env = PixelWrapper(cfg, env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	
	return env
