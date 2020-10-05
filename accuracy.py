import torch
from utils import * 
import numpy as np

def accuracy(k,j,edges_number_list,output_final,ground_truth,batch,start,device):
    num_of_edges= edges_number_list[int(j.item())]
    output3= [0] * num_of_edges
    output_sliced= output_final[start:start+num_of_edges].detach().clone()
    ground_truth_sliced= ground_truth[start:start+num_of_edges].to(torch.int8).detach().clone()
    edges_list_reduced= [[],[]]
    output_reduced= []
    ground_truth_reduced= []
    # print(j.item())
    for i in range(len(batch[k].edge_index[0])):
        edge1= batch[k].edge_index[0][i]
        edge2= batch[k].edge_index[1][i]
        if edge1<=edge2:
            edges_list_reduced[0].append(edge1.item())
            edges_list_reduced[1].append(edge2.item())
            ground_truth_reduced.append(ground_truth_sliced[i])
            if edge1<edge2: #find the second same edge
                for j in range(i,len(batch[k].edge_index[0])):
                    edge3= batch[k].edge_index[0][j]
                    edge4= batch[k].edge_index[1][j]
                    if edge1==edge4 and edge2==edge3:
                        output_reduced.append((output_sliced[i]+output_sliced[j])/2)
                        break
            else:
                output_reduced.append(output_sliced[i])
    start += num_of_edges
    constraints= []
    # find indexes for constraints
    max= 0
    for i in range(len(batch[k].edge_index[0])):
        out1 = indices_first(edges_list_reduced[0], edges_list_reduced[1], i)
        out2 = indices_second(edges_list_reduced[0], edges_list_reduced[1], i)
        if out1 and len(np.array(out1))>1:
            if len(out1)>max:
                max= len(out1)
            constraints.append(out1)
        if out2 and len(np.array(out2))>1:
            if len(out2)>max:
                max= len(out2)
            constraints.append(out2)
    # Get most probable edges as 1 and the other as 0
    max=0
    zero_indeces= []
    one_indeces= []
    ranking= True
    # print(optim_graph.out.size())
    while ranking==True:
        for i, edge in enumerate(output_reduced):
            if (edge>max) and (i not in zero_indeces) and (i not in one_indeces):
                max=edge
                index= i
        if max==0:
            ranking= False
        
        else:
            one_indeces.append(index)
            for constraint in constraints:
                if index in constraint:
                    for constr in constraint:
                        if constr!=index and constr!=-1 and constr not in zero_indeces:
                            zero_indeces.append(constr)    
            max=0
    processed_output= []
    for i, edge in enumerate(output_reduced):
        if i in one_indeces:
            processed_output.append(torch.tensor(1).to(device))
        else:
            processed_output.append(torch.tensor(0).to(device))
    processed_output= torch.stack(processed_output)
    ground_truth_reduced= torch.stack(ground_truth_reduced)
    
    return processed_output, ground_truth_reduced,output_reduced