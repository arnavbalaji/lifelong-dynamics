import os
import torch
import numpy as np
import cv2
import h5py

from libero.libero import benchmark, get_libero_path, set_libero_default_path
from libero.libero.envs import OffScreenRenderEnv

set_libero_default_path("/home/arpit/projects/tdmpc2/LIBERO/libero/libero")

benchmark_dict = benchmark.get_benchmark_dict()
print(benchmark_dict)
task_suite_name = "libero_spatial" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

print(task_suite)

task_id = 0
# while True:
#     try:
#         task = task_suite.get_task(task_id)
#         task_name = task.name
#         print(f"\nTask name: {task_name}\n")
#     except:
#         break
#     task_id += 1

task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name


# for filename in os.listdir("LIBERO/libero/datasets/libero_spatial"):
#     if filename.endswith(".hdf5"):
#         print(filename)
#         file = h5py.File(os.path.join("LIBERO/libero/datasets/libero_spatial", filename))
#         data = file["data"]
#         for d in data.items():
#             cv2.destroyAllWindows()
#             demo_name = d[0]
#             demo = d[1]
#             rgb_images = np.array(demo["obs"]["agentview_rgb"])
#             for img in rgb_images:
#                 flipped_image = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
#                 resized_image = cv2.resize(flipped_image, (512, 512), interpolation=cv2.INTER_AREA)
#                 cv2.imshow(demo_name, resized_image)
#                 cv2.waitKey(20)
#             break

file = h5py.File(f"LIBERO/libero/datasets/libero_spatial/{task_name}_demo.hdf5")
print(f"LIBERO/libero/datasets/libero_spatial/{task_name}_demo.hdf5")

print(list(file.items()))
data = file["data"]
# breakpoint()
# quit()
# for d in data.items():
#     cv2.destroyAllWindows()
#     demo_name = d[0]
#     demo = d[1]
#     rgb_images = np.array(demo["obs"]["agentview_rgb"])
#     for img in rgb_images:
#         flipped_image = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
#         resized_image = cv2.resize(flipped_image, (512, 512), interpolation=cv2.INTER_AREA)
#         cv2.imshow(demo_name, resized_image)
#         cv2.waitKey(20)
# breakpoint()
# quit()

actions = np.array(list(data.items())[0][1]["actions"])

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
print(f"\nTask name: {task_name}\n")
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
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
# for i in range(init_states.shape[0]):
#     init_state_id = i
#     env.set_init_state(init_states[init_state_id])

#     dummy_action = [0.] * 7
#     for step in range(10):
#         obs, reward, done, info = env.step(dummy_action)
#         # flipped_image = cv2.flip(cv2.cvtColor(obs["agentview_image"], cv2.COLOR_BGR2RGB), 0)
#         # resized_image = cv2.resize(flipped_image, (512, 512), interpolation=cv2.INTER_AREA)
#         # cv2.imshow('Images', resized_image)
#         # cv2.waitKey(0)

#     flipped_image = cv2.flip(cv2.cvtColor(obs["agentview_image"], cv2.COLOR_BGR2RGB), 0)
#     resized_image = cv2.resize(flipped_image, (512, 512), interpolation=cv2.INTER_AREA)
#     cv2.imshow('Images', resized_image)
#     cv2.waitKey(10)

init_state_id = 0
env.set_init_state(init_states[init_state_id])
data_items = sorted(list(data.items()), key=lambda x: int(x[0].split('_')[1]))
for init_state_id in range(50):
    cv2.destroyAllWindows()
    env.set_init_state(init_states[init_state_id])
    actions = np.array(data_items[init_state_id][1]["actions"])
    for i in range(10):
        env.reset()
        dummy_action = [0.] * 7
        for step in range(10):
            obs, reward, done, info = env.step(dummy_action)

        for step in range(actions.shape[0]):
            action = actions[step] + np.random.normal(0, 0.1, actions[step].shape)
            obs, reward, done, info = env.step(action)
            # breakpoint()
            # quit()
            flipped_image = cv2.flip(cv2.cvtColor(obs["agentview_image"], cv2.COLOR_BGR2RGB), 0)
            resized_image = cv2.resize(flipped_image, (512, 512), interpolation=cv2.INTER_AREA)
            cv2.imshow(f'{init_state_id}', resized_image)
            cv2.waitKey(10)
        if reward == 1:
            print("\nSUCCESS\n")
        else:
            print("\nFAILURE\n")

cv2.destroyAllWindows()
env.close()