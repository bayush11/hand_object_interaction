import os
import random
from utils.hand_model_lite import HandModelMJCFLite
import numpy as np
import transforms3d
import torch
import trimesh

mesh_path = "../data/meshdata"
data_path = "../data/dataset"

use_visual_mesh = False

hand_file = "mjcf/shadow_hand_vis.xml" if use_visual_mesh else "mjcf/shadow_hand_wrist_free.xml"

joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]
translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

hand_model = HandModelMJCFLite(
    hand_file,
    "mjcf/meshes")

grasp_code_list = []
for code in os.listdir(data_path):
    grasp_code_list.append(code[:-4])

# grasp_code = random.choice(grasp_code_list)
grasp_code = grasp_code_list[0]
grasp_data = np.load(
    os.path.join(data_path, grasp_code+".npy"), allow_pickle=True)
object_mesh_origin = trimesh.load(os.path.join(
    mesh_path, grasp_code, "coacd/decomposed.obj"))
print(len(grasp_code_list))
print(len(grasp_data))
print(grasp_code)

index = random.randint(0, len(grasp_data) - 1)


qpos = grasp_data[index]['qpos']
rot = np.array(transforms3d.euler.euler2mat(
    *[qpos[name] for name in rot_names]))
rot = rot[:, :2].T.ravel().tolist()
hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name]
                         for name in joint_names], dtype=torch.float, device="cpu").unsqueeze(0)
print(len(joint_names))

    

hand_pose = torch.tensor([ 0,  0, 0,  0,  0, -.7245e-01, 0,  -8.8080e-01,  0, # wrist
                0, 0.1357273946202482, 0.0326704164339473, -0.0783459355437075, # pointer
                0, 0.24203054348575037, -0.010857365529401086, -0.10094950695865501, # middle
                0, 0.3511986215902736, -0.20004214604748327, 0.06653641197372606,  # ring
                0, 0, 0.2301150341822078, 0.26521111766444605, -0.2635448066396169, # pinky
                0, 0, -0.17645134031772614, -0.04307592287659645, 0.09345488995313644], dtype=torch.float, device="cpu").unsqueeze(0) # thumb

hand_pose = torch.tensor([ 0,  0, 0,  0,  0, -.7245e-01, 0,  -8.8080e-01,  0, # wrist
               0, 0.12269129256374489, 1.0016951789452007, 0.455554481843778, 0, 0.5270930428300993, 0.9854085971905162, 0.4487033079064573, 0, 0.6487723370386531, 0.42097428182708185, 0.6676155183045976, 0, 0, 0.4340308388067653, 0.3925107393656866, -0.028361955086653623, 0, 0, 0.156825453042984, -0.006430144887417555, 0.6274697184562683], dtype=torch.float, device="cpu").unsqueeze(0) # thumb
# print(hand_pose.shape)
hand_model.set_parameters(hand_pose)
hand_mesh = hand_model.get_trimesh_data(0)
object_mesh = object_mesh_origin.copy().apply_scale(grasp_data[index]["scale"])

(hand_mesh).show()