from torch_geometric.data import DataLoader, DataListLoader
from build_graph import *
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import torchvision
import torchvision.utils as vutils
import shutil
import tensorflow as tf
import tensorboard as tb
import keyword
from torchvision.transforms import ToTensor
from utils import *
import datetime

def model_testing(sequence, detections, images_path, total_frames, frames_look_back, model, distance_limit, fp_min_times_seen, match_thres, det_conf_thres, fp_look_back, fp_recent_frame_limit,min_height,fps):

    device = torch.device('cuda')
    #pick one frame and load previous results
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
    current_frame= 2
    id_num= 0
    tracking_output= []
    checked_ids = []
    
    transform = ToTensor()

    while current_frame <= total_frames:
        print("Sequence: " + sequence+ ", Frame: " + str(current_frame)+'/'+str(int(total_frames)))
        data_list = []
        #Give IDs to the first frame
        tracklets = []
        if not tracking_output:
            for i, detection in enumerate(detections):
                if detection[0] == 1:
                    frame = detection[0]
                    xmin, ymin, width, height = int(round(detection[2])), int(round(detection[3])), \
                                                int(round(detection[4])), int(round(detection[5]))
                    confidence= detection[6]
                    if xmin > 0 and ymin > 0 and width > 0 and height > min_height and confidence>det_conf_thres:
                        id_num += 1
                        ID= int(id_num)
                        tracking_output.append([frame, ID, xmin, ymin, width, height, \
                            int(detection[6]), 1, 1])
                        tracklets.append([[frame, ID, xmin, ymin, width, height, \
                                                int(detection[6]), 1, 1]])
                else:
                    detections= detections[i:]
                    break
        else:
            #Get all tracklets
            tracklet_IDs = []
            for j, tracklet in enumerate(tracking_output):
                xmin, ymin, width, height = int(round(tracklet[2])), int(round(tracklet[3])), \
                                            int(round(tracklet[4])), int(round(tracklet[5]))
                if xmin > 0 and ymin > 0 and width > 0 and height > 0:
                    if (tracklet[0]<current_frame) and tracklet[0]>=current_frame-frames_look_back:
                        new_tracklet= True
                        for k,i in enumerate(tracklet_IDs):
                            if tracklet[1]==i:
                                new_tracklet=False
                                tracklets[k].append(tracklet)
                                break
                        if new_tracklet==True:
                            tracklet_IDs.append(int(tracklet[1]))
                            tracklets.append([tracklet])
        #Get new detections
        current_detections = []
        for i, detection in enumerate(detections):
            if detection[0] == current_frame:
                frame = detection[0]
                xmin, ymin, width, height = int(round(detection[2])), int(round(detection[3])), \
                                            int(round(detection[4])), int(round(detection[5]))
                confidence= detection[6]
                if xmin > 0 and ymin > 0 and width > 0 and height > min_height and confidence>det_conf_thres:
                    current_detections.append([frame, -1, xmin, ymin, width, height, \
                        int(detection[6]), 1, 1])
            else:
                detections= detections[i:]
                break
        #build graph and run model
        data = build_graph(tracklets, current_detections, images_path, current_frame, distance_limit, fps, test=True)
        if data:
            if current_detections and data.edge_attr.size()[0]!=0:
                data_list.append(data)

                loader = DataListLoader(data_list)
                for graph_num, batch in enumerate(loader):
                    #MODEL FORWARD
                    output, output2, ground_truth, ground_truth2, det_num, tracklet_num= model(batch)
                    #FEATURE MAPS on tensorboard
                    #embedding
                    images= batch[0].x
                    images = F.interpolate(images, size=250)
                    edge_index= data_list[graph_num].edges_complete
                    #THRESHOLDS
                    temp= []
                    for i in output2:
                        if i>match_thres:
                            temp.append(i)
                        else:
                            temp.append(i-i)
                    output2= torch.stack(temp)
                    # HUNGARIAN
                    cleaned_output= hungarian(output2, ground_truth2, det_num, tracklet_num)
                    # Give Ids to current frame
                    for i,detection in enumerate(current_detections):
                        match_found= False
                        for k,m in enumerate(cleaned_output):#cleaned_output):
                            if m==1 and edge_index[1,k]==i+len(tracklets): #match found
                                ID= tracklets[edge_index[0,k]][-1][1]
                                frame = detection[0]
                                xmin, ymin, width, height = int(round(detection[2])), int(round(detection[3])), \
                                                            int(round(detection[4])), int(round(detection[5]))
                                tracking_output.append([frame, ID, xmin, ymin, width, height, \
                                                        int(detection[6]), 1, 1])
                                match_found = True
                                break
                        if match_found==False: #give new ID
                            # print("no match")
                            id_num += 1
                            ID= id_num
                            frame = detection[0]
                            xmin, ymin, width, height = int(round(detection[2])), int(round(detection[3])), \
                                                        int(round(detection[4])), int(round(detection[5]))
                            tracking_output.append([frame, ID, xmin, ymin, width, height, \
                                                    int(detection[6]), 1, 1])
                    #Clean output for false positives
                    if current_frame>=fp_look_back:
                        # reduce to recent objects
                        recent_tracks = [i for i in tracking_output if i[0] >= current_frame-fp_look_back]
                        # find the different IDs
                        candidate_ids= []
                        times_seen= []
                        first_frame_seen= []
                        for i in recent_tracks:
                            if i[1] not in checked_ids:
                                if i[1] not in candidate_ids:
                                    candidate_ids.append(i[1])
                                    times_seen.append(1)
                                    first_frame_seen.append(i[0])
                                else:
                                    index= candidate_ids.index(i[1])
                                    times_seen[index]= times_seen[index] + 1
                        # find which IDs to remove
                        remove_ids = []
                        for i,j in enumerate(candidate_ids):
                            if times_seen[i] < fp_min_times_seen and current_frame-first_frame_seen[i]>=fp_look_back:
                                remove_ids.append(j)
                            elif times_seen[i] > fp_min_times_seen:
                                checked_ids.append(j)
                        #keep only those IDs that are seen enough times
                        tracking_output = [j for j in tracking_output if j[1] not in remove_ids]
        current_frame += 1
    # reduce to recent objects
    recent_tracks = [i for i in tracking_output if i[0] >= current_frame-fp_look_back]
    # find the different IDs
    candidate_ids= []
    times_seen= []
    for i in recent_tracks:
        if i[1] not in checked_ids:
            if i[1] not in candidate_ids:
                candidate_ids.append(i[1])
                times_seen.append(1)
            else:
                index= candidate_ids.index(i[1])
                times_seen[index]= times_seen[index] + 1
    # find which IDs to remove
    remove_ids = []
    for i,j in enumerate(candidate_ids):
        if times_seen[i] < fp_min_times_seen:
            remove_ids.append(j)
        elif times_seen[i] > fp_min_times_seen:
            checked_ids.append(j)
    #keep only those IDs that are seen enough times
    tracking_output = [j for j in tracking_output if j[1] not in remove_ids]

    return tracking_output