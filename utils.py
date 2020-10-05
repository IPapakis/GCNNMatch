import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def fill_detections(tracking_output):
    final_output= []
    different_ids= []
    for i in tracking_output:
        if i[1] not in different_ids:
            different_ids.append(int(i[1]))
    #for each ID
    for i in different_ids:
        #get output for each ID
        output_temp= []
        for j in tracking_output:
            if j[1]==i:
                output_temp.append(j)
        filled_detections= []
        #for every two frames, fill in detections
        for j in range(len(output_temp)-1):
            diff_frame= output_temp[j+1][0]-output_temp[j][0]
            diff_x= output_temp[j][2]-output_temp[j+1][2]
            diff_y= output_temp[j][3]-output_temp[j+1][3]
            diff_w= output_temp[j][4]-output_temp[j+1][4]
            diff_h= output_temp[j][5]-output_temp[j+1][5]
            boxes1= torch.tensor([output_temp[j][2],output_temp[j][3],output_temp[j][2]+output_temp[j][4],output_temp[j][3]+output_temp[j][5]])
            boxes2= torch.tensor([output_temp[j+1][2],output_temp[j+1][3],output_temp[j+1][2]+output_temp[j+1][4],output_temp[j+1][3]+output_temp[j+1][5]])
            IOU= box_iou_calc(boxes1, boxes2)
            if abs(int(diff_frame))>1 and abs(int(diff_frame))<150 and IOU.item()>0.1:                  
                for sec in range(1,abs(int(diff_frame))):
                    div= abs(diff_frame)/sec
                    filled_detections.append([output_temp[j][0]+sec,i,int(output_temp[j][2]-diff_x/div),\
                        int(output_temp[j][3]-diff_y/div),int(output_temp[j][4]-diff_w/div),int(output_temp[j][5]-diff_h/div)])
        for j in output_temp:
            final_output.append(j)
        for j in filled_detections:
            final_output.append(j)
    final_output = sorted(final_output, key=lambda x: x[0])
    return final_output


def weighted_binary_cross_entropy(output, target, weights=None):
    loss = - weights[0] * (target * torch.log(output)) - \
            weights[1] * ((1 - target) * torch.log(1 - output))
    return torch.mean(loss)

def indices_first(a, b, value):
    out = [k for k, x in enumerate(a) if x == value]# and x <= b[k]]
    if out:
        return out

def indices_second(a, b, value):
    out = [k for k, x in enumerate(b) if x == value]# and x >= a[k]]
    if out:
        return out

def sinkhorn(matrix): 
    row_len = len(matrix) 
    col_len = len(matrix[0]) 
    desired_row_sums = torch.ones((1, row_len), requires_grad=False).cuda()
    desired_col_sums = torch.ones((1, col_len), requires_grad=False).cuda()
    desired_row_sums[:, -1] = col_len-1
    desired_col_sums[:, -1] = row_len-1
    for _ in range(8):
        #row normalization
        actual_row_sum = torch.sum(matrix, axis=1)
        for i, row in enumerate(matrix):
            for j, element in enumerate(row):
                matrix[i,j]= element*desired_row_sums[0,i]/(actual_row_sum[i])
        #column normalization
        actual_col_sum = torch.sum(matrix, axis=0)
        for i, row in enumerate(matrix):
            for j, element in enumerate(row):
                matrix[i,j]= element*desired_col_sums[0,j]/(actual_col_sum[j])
    return matrix


def hungarian(output, ground_truth, det_num, tracklet_num):
    cleaned_output = []
    num = 0
    eps = 0.0001  # for numerical stability
    for i, j in enumerate(tracklet_num):
        matrix = []
        for k in range(j):
            matrix.append([])
            for l in range(det_num[i]):
                matrix[k].append(1 - output[num].cpu().detach().numpy())
                num += 1
        matrix = np.array(matrix)
        # padding
        (a, b) = matrix.shape
        if a > b:
            padding = ((0, 0), (0, a - b))
        else:
            padding = ((0, b - a), (0, 0))
        matrix = np.pad(matrix, padding, mode='constant', constant_values=eps)
        # hungarian
        row_ind, col_ind = linear_sum_assignment(matrix)
        #take out those that are all 1, max cost, either hungarian will assign
        remove_ind= []
        cnt= 0
        for i, row in enumerate(matrix):
            for j, element in enumerate(row):
                if element==1:
                    remove_ind.append(cnt)
                cnt += 1
        cnt= 0
        for i, row in enumerate(matrix):
            for j, element in enumerate(row):
                if i < a and j < b:
                    p1 = row_ind.tolist().index(i)
                    p2 = col_ind.tolist().index(j)
                    # print(p2)
                    if p1 == p2 and cnt not in remove_ind:
                        cleaned_output.append(torch.tensor(1, dtype=float).cuda())
                    else:
                        cleaned_output.append(torch.tensor(0, dtype=float).cuda())
                cnt += 1
    cleaned_output = torch.stack(cleaned_output)
    return cleaned_output

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    boxes1= boxes1.reshape(1,4)
    boxes2 = boxes2.reshape(1, 4)

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor