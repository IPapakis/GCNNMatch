import numpy as np
import torch
from PIL import Image
from utils import *
import torch.nn.functional as F
from utils import * 
from torch_geometric.data import Data, DataLoader, DataListLoader
from random import randint
from build_graph import *

def data_prep_train(sequence, detections, images_path, frames_look_back, total_frames, most_recent_frame_back, graph_jump, current_frame_train, current_frame_valid, distance_limit, fps, type):

    if total_frames==None:
        total_frames= np.max(detections[:,0]) #change only if you want a subset of the total frames
    detections= sorted(detections, key = lambda x: x[0])
    data_list = []
    acceptable_object_types= [1,2,7] # MOT specific types
    if type=="training":
        total_frames= current_frame_valid
        current_frame= current_frame_train
    elif type=="validation":
        total_frames= total_frames
        current_frame= current_frame_valid

    while current_frame<=total_frames:

        print("Sequence: " + sequence + ", Frame: " + str(current_frame)+"/"+ str(int(total_frames)))
        ####find tracklets and new detections
        current_detections = []
        tracklets = []
        tracklet_IDs = []
        for j, detection in enumerate(detections):
            if detection[0]>current_frame:
                break
            else:
                xmin, ymin, width, height = int(round(detection[2])), int(round(detection[3])), \
                                            int(round(detection[4])), int(round(detection[5]))
                object_type = detection[7]
                if xmin > 0 and ymin > 0 and width > 0 and height > 0 and (object_type in acceptable_object_types):
                    most_recent_frame_back2 = randint(1, most_recent_frame_back)
                    if current_frame-most_recent_frame_back2<1:
                        most_recent_frame_back2=1
                    temp=current_frame - (most_recent_frame_back2 - 1)
                    if (detection[0]<temp) and detection[0]>=temp-frames_look_back:
                        new_tracklet= True
                        for k,i in enumerate(tracklet_IDs):
                            if detection[1]==i:
                                new_tracklet=False
                                tracklets[k].append(detection)
                                break
                        if new_tracklet==True:
                            tracklet_IDs.append(int(detection[1]))
                            tracklets.append([detection])
                    elif detection[0]==current_frame:
                        current_detections.append(detection)
        data = build_graph(tracklets, current_detections, images_path, current_frame, distance_limit, fps, test=False)
        data_list.append(data)
        current_frame += graph_jump
    print("Data preparation finished")
    return data_list


