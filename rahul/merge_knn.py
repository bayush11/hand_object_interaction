import json
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

def convert_mano_to_shadowhand():
    print("Converting MANO to Shadowhand...")
    tf_matrix = np.array([
        -np.pi/8, # pointer finger lower
        -np.pi/4,   # pointer finger middle
        -np.pi/8,  # pointer finger upper
        -np.pi/5, # middle finger lower
        -np.pi/4,  # middle finger middle
        -np.pi/8,  # middle finger upper
        -np.pi/4,  # pinky finger lower
        -np.pi/5,  # pinky finger middle
        -np.pi/10,  # pinky finger upper
        -np.pi/4,  # ring finger lower
        -np.pi/5,  # ring finger middle
        -np.pi/6, # ring finger upper
        0, # thumb finger lower
        0, # thumb finger middle
        0,  # thumb finger upper
    ])

    mano = json.load(open("fhand/data/FreiHAND_pub_v2/training_mano.json"))
    shadowhand = []
    for i in mano:
        mano_joints = np.array(i[0][3:48][2::3])
        mano_joints = np.array(mano_joints)-np.array(tf_matrix)
        mano_joints = mano_joints.tolist()
        shadowhand_joints = [0] + mano_joints[0:3] + [0] + mano_joints[3:6] + [0] + mano_joints[9:12] + [0,0] + mano_joints[6:9] + [0,0] + mano_joints[12:]
        shadowhand.append(shadowhand_joints)

    with open('data.json', 'w') as f:
        json.dump(shadowhand, f)
        
def convert_dgnet_to_training():
    print("Converting DG-Net to training data...")
    joint_names = [
        'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    ]
    
    grasp_code_list = []
    for code in os.listdir("dgnet/data/dexgraspnet"):
        grasp_code_list.append(code[:-4])
        
    print(str(len(grasp_code_list)) + " grasps found.")
    
    grasp_dict = {}

    for i in grasp_code_list:
        count = 0
        for j in np.load(
            os.path.join("dgnet/data/dexgraspnet", i+".npy"), allow_pickle=True):
            count+=1
            grasp_dict[i+"_"+str(count)] = [j['qpos'][name] for name in joint_names]
    
    with open("training.json", "w") as f:
        json.dump(grasp_dict, f)


def perform_knn():

    print("Retrieving data from FreiHAND dataset...")

    shadowhand_test = json.load(open("data.json"))

    print("Retrieving data from DG-Net dataset...")

    grasp_dict = json.load(open("training.json"))

    print("Training KNN...")

    grasp_dict_values = list(grasp_dict.values())

    # Convert lists to numpy arrays for processing
    fhand_np = np.array(shadowhand_test)
    dgnet_np = np.array(grasp_dict_values)  # Flattening list2

    print(fhand_np.shape)
    print(dgnet_np.shape)
    # Initialize the NearestNeighbors class with n_neighbors as 100
    neigh = NearestNeighbors(n_neighbors=100)

    # Fit the model using list2 (the neighbors)
    neigh.fit(dgnet_np)

    print("Evaluating KNN...")
    distance_values = []
    index_values = []
    # For each data point in list1, find the 100 nearest neighbors from list2
    for datapoint in fhand_np:
        distances, indices = neigh.kneighbors([datapoint])
        distance_values.append(distances[0].tolist())
        index_values.append(indices[0].tolist())


    # Save the results to a JSON file
    with open("results.json", "w") as f:
        json.dump({"distances": distance_values, "indices": index_values}, f)
        
    print("SUCCESS!")
    
def read_results():
    results = json.load(open("results.json"))
    distances = results["distances"]
    indices = results["indices"]
    dgnet_dict = json.load(open("training.json"))
    fhand_dict = json.load(open("data.json"))
        
if __name__ == "__main__":
    # convert_mano_to_shadowhand()
    # convert_dgnet_to_training()
    # perform_knn()
    read_results()
    pass



