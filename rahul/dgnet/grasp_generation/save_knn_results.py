import os
import random
from utils.hand_model_lite import HandModelMJCFLite
import numpy as np
import transforms3d
import torch
import trimesh
import json
from PIL import Image
import io

# loads all the assets required for the hand model
mesh_path = "../data/meshdata"
data_path = "../data/dexgraspnet"

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
    "mjcf/meshes", device="cuda")

def load_data():
    results = json.load(open("../../results.json"))
    # distances = results["distances"]
    indices = results["indices"]
    dgnet_dict = json.load(open("../../training.json"))
    fhand_dict = json.load(open("../../data.json"))
    dgnet_keys = list(dgnet_dict)
    return dgnet_keys, indices, fhand_dict

def view_fhand(index, fhand_dict):
    rot = [0,  0, 0,  0,  0, -.7245e-01, 0,  -8.8080e-01,  0,]
    hand_pose = torch.tensor(rot + fhand_dict[index], dtype=torch.float, device="cuda").unsqueeze(0)
    hand_model.set_parameters(hand_pose)
    hand_mesh = hand_model.get_trimesh_data(0)
    (hand_mesh).show()

def save_img(grasp_code, index, directory, fhand_data):
    grasp_data = np.load(
        os.path.join(data_path, grasp_code+".npy"), allow_pickle=True)

    qpos = grasp_data[index]['qpos']
    rot = np.array(transforms3d.euler.euler2mat(
        *[qpos[name] for name in rot_names]))
    rot = rot[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name]
                            for name in joint_names], dtype=torch.float, device="cuda").unsqueeze(0)
    hand_model.set_parameters(hand_pose)
    hand_mesh = hand_model.get_trimesh_data(0)
    scene = hand_mesh.scene()
    image = scene.save_image(resolution=[640, 480], visible=True)
    image_data_io = io.BytesIO(image)
    pil_image = Image.open(image_data_io)
    pil_image.save(f"{directory}dgnet/{grasp_code}_{index}.png")   
    
    hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + fhand_data, dtype=torch.float, device="cuda").unsqueeze(0) 
    hand_model.set_parameters(hand_pose)
    hand_mesh = hand_model.get_trimesh_data(0)
    scene = hand_mesh.scene()
    image = scene.save_image(resolution=[640, 480], visible=True)
    image_data_io = io.BytesIO(image)
    pil_image = Image.open(image_data_io)
    pil_image.save(f"{directory}fhand/original_{grasp_code}_{index}.png")   

def save_all_imgs(dgnet_keys, indices, fhand_dict, num_to_save):
    for i in range(0, num_to_save):
        print("Saving: ", i, " out of ", num_to_save, " images")
        os.makedirs("knn_results_visuals/"+str(i)+"/dgnet", exist_ok=True)
        os.makedirs("knn_results_visuals/"+str(i)+"/fhand", exist_ok=True)
        for j in range(0, 10):
            split = dgnet_keys[indices[i][j]].rsplit("_", 1)
            save_img(split[0], int(split[1])-1, "knn_results_visuals/"+str(i) + "/", fhand_dict[i])
            
if __name__ == "__main__":
    dgnet_keys, indices, fhand_dict = load_data()
    # view_fhand(3, json.load(open("../../data.json")))
    save_all_imgs(dgnet_keys, indices, fhand_dict, 100)