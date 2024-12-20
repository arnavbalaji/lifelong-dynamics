class Trainer:
	"""Base trainer class for TD-MPC2."""

	def __init__(self, cfg, env, agent, buffer, logger, show_images=False, offline_buffer=None):
		self.cfg = cfg
		self.env = env
		self.agent = agent
		self.buffer = buffer
		self.logger = logger
		self.show_images = show_images
		self.offline_buffer = offline_buffer
		print('Architecture:', self.agent.model)
		print("Learnable parameters: {:,}".format(self.agent.model.total_params))

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		raise NotImplementedError

	def train(self):
		"""Train a TD-MPC2 agent."""
		raise NotImplementedError
