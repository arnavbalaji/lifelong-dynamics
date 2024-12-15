import os
import torch
import numpy as np
import cv2
import h5py
import sys
import time

from libero.libero import benchmark, get_libero_path, set_libero_default_path
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()
print(benchmark_dict)
task_suite_name = "libero_spatial" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

file = h5py.File(f"LIBERO/libero/datasets/libero_spatial/{task_name}_demo.hdf5")
# print(f"LIBERO/libero/datasets/libero_spatial/{task_name}_demo.hdf5")
data = file["data"]

dataset_filename = f"LIBERO/libero/datasets/libero_spatial/{task_name}_extra_data_5000.hdf5"
print(f"Generating dataset to {dataset_filename}")

def print_progress_bar(iteration, total, successes, failures, length=40):
    percent = (iteration / total)
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent:.2%}')
    sys.stdout.write(f"   Num successes: {successes}")
    sys.stdout.write(f"   Num failures: {failures}")
    sys.stdout.flush()

with h5py.File(dataset_filename, 'w') as f:
    file_data = f.create_group("data")

env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 512,
    "camera_widths": 512
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
print(f"\n\nNumber of init states: {init_states.shape[0]}\n")

num_success = 0
num_failures = 0

print("\n\nStarting Data Generation\n")
data_items = sorted(list(data.items()), key=lambda x: int(x[0].split('_')[1]))
for init_state_id in range(50):
    cv2.destroyAllWindows()
    env.reset()
    env.set_init_state(init_states[init_state_id])
    actions = np.array(data_items[init_state_id][1]["actions"])
    for i in range(100):
        obs_list = []
        rewards_list = []
        dones_list = []
        actions_list = []

        env.reset()
        dummy_action = [0.] * 7
        for step in range(20):
            obs, reward, done, info = env.step(dummy_action)

        obs_list.append(obs["agentview_image"])

        for step in range(actions.shape[0]):
            action = actions[step] + np.random.normal(0, 0.115, actions[step].shape)
            obs, reward, done, info = env.step(action)
            obs_list.append(obs["agentview_image"])
            rewards_list.append(reward)
            dones_list.append(done)
            actions_list.append(action)

            # flipped_image = cv2.flip(cv2.cvtColor(obs["agentview_image"], cv2.COLOR_BGR2RGB), 0)
            # resized_image = cv2.resize(flipped_image, (512, 512), interpolation=cv2.INTER_AREA)
            # cv2.imshow(f'{init_state_id}', resized_image)
            # cv2.waitKey(10)
        if reward == 1:
            # print("SUCCESS")
            num_success += 1
        else:
            # print("FAILURE")
            num_failures += 1
        
        # Add demos to HDF5 file here
        # if reward == 1:
        with h5py.File(dataset_filename, "r+") as file:
            data = file["data"]
            demo = data.create_group(f"demo_{(init_state_id * 200) + i}")
            obs_group = demo.create_group("obs")
            agentview_rgb_dataset = obs_group.create_dataset('agentview_rgb', 
                                        data=np.array(obs_list),
                                        compression='gzip',
                                        compression_opts=9)

            reward_dataset = demo.create_dataset('rewards', 
                                        data=np.array(rewards_list),
                                        compression='gzip',
                                        compression_opts=9)
            dones_dataset = demo.create_dataset('dones', 
                                        data=dones_list,
                                        compression='gzip',
                                        compression_opts=9)
            actions_dataset = demo.create_dataset('actions', 
                                        data=actions_list,
                                        compression='gzip',
                                        compression_opts=9)
        print_progress_bar((init_state_id * 200) + i + 1, 50 * 200, num_success, num_failures)

print(f"\n\nNumber of successes: {num_success}")
print(f"Number of failures: {num_failures}")
print(f"Success rate: {num_success / (num_success + num_failures)}")

cv2.destroyAllWindows()
env.close()