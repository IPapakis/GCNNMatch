from torch_geometric.nn import MetaLayer, DataParallel
from utils import * 
from network.complete_net import *
from torch_geometric.data import DataLoader, DataListLoader
from accuracy import *
import matplotlib.pyplot as plt
import logging
import sys
import os

def model_training(data_list_train, data_list_test, epochs, acc_epoch, acc_epoch2, save_model_epochs, validation_epoch, batchsize, logfilename, load_checkpoint= None):
        
    #logging
    logging.basicConfig(level=logging.DEBUG, filename='./logfiles/'+logfilename, filemode="w+",
                        format="%(message)s")
    trainloader = DataListLoader(data_list_train, batch_size=batchsize, shuffle=True)
    testloader = DataListLoader(data_list_test, batch_size=batchsize, shuffle=True)
    device = torch.device('cuda')
    complete_net = completeNet()
    complete_net = DataParallel(complete_net)
    complete_net = complete_net.to(device)
    
    #train parameters
    weights = [10, 1]
    optimizer = torch.optim.Adam(complete_net.parameters(), lr=0.001, weight_decay=0.001)

    #resume training
    initial_epoch=1
    if load_checkpoint!=None:
        checkpoint = torch.load(load_checkpoint)
        complete_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch']+1
        loss = checkpoint['loss']
    
    complete_net.train()

    for epoch in range(initial_epoch, epochs+1):
        epoch_total=0
        epoch_total_ones= 0
        epoch_total_zeros= 0
        epoch_correct=0
        epoch_correct_ones= 0
        epoch_correct_zeros= 0
        running_loss= 0
        batches_num=0         
        for batch in trainloader:
            batch_total=0
            batch_total_ones= 0
            batch_total_zeros= 0
            batch_correct= 0
            batch_correct_ones= 0
            batch_correct_zeros= 0
            batches_num+=1
            # Forward-Backpropagation
            output, output2, ground_truth, ground_truth2, det_num, tracklet_num= complete_net(batch)
            optimizer.zero_grad()
            loss = weighted_binary_cross_entropy(output, ground_truth, weights)
            loss.backward()
            optimizer.step()
            ##Accuracy 
            if epoch%acc_epoch==0 and epoch!=0:
                # Hungarian method, clean up
                cleaned_output= hungarian(output2, ground_truth2, det_num, tracklet_num)
                batch_total += cleaned_output.size(0)
                ones= torch.tensor([1 for x in cleaned_output]).to(device)
                zeros = torch.tensor([0 for x in cleaned_output]).to(device)
                batch_total_ones += (cleaned_output == ones).sum().item()
                batch_total_zeros += (cleaned_output == zeros).sum().item()
                batch_correct += (cleaned_output == ground_truth2).sum().item()
                temp1 = (cleaned_output == ground_truth2)
                temp2 = (cleaned_output == ones)
                batch_correct_ones += (temp1 & temp2).sum().item()
                temp3 = (cleaned_output == zeros)
                batch_correct_zeros += (temp1 & temp3).sum().item()
                epoch_total += batch_total
                epoch_total_ones += batch_total_ones
                epoch_total_zeros += batch_total_zeros
                epoch_correct += batch_correct
                epoch_correct_ones += batch_correct_ones
                epoch_correct_zeros += batch_correct_zeros
            if loss.item()!=loss.item():
                print("Error")
                break
            if batch_total_ones != 0 and batch_total_zeros != 0 and epoch%acc_epoch==0 and epoch!=0:
                print('Epoch: [%d] | Batch: [%d] | Training_Loss: %.3f | Total_Accuracy: %.3f | Ones_Accuracy: %.3f | Zeros_Accuracy: %.3f |' %
                      (epoch, batches_num, loss.item(), 100 * batch_correct / batch_total, 100 * batch_correct_ones / batch_total_ones,
                       100 * batch_correct_zeros / batch_total_zeros))
                logging.info('Epoch: [%d] | Batch: [%d] | Training_Loss: %.3f | Total_Accuracy: %.3f | Ones_Accuracy: %.3f | Zeros_Accuracy: %.3f |' %
                      (epoch, batches_num, loss.item(), 100 * batch_correct / batch_total, 100 * batch_correct_ones / batch_total_ones,
                       100 * batch_correct_zeros / batch_total_zeros))
            else:
                print('Epoch: [%d] | Batch: [%d] | Training_Loss: %.3f |' %
                        (epoch, batches_num, loss.item()))
                logging.info('Epoch: [%d] | Batch: [%d] | Training_Loss: %.3f |' %
                        (epoch, batches_num, loss.item()))
            running_loss += loss.item()
        if loss.item()!=loss.item():
                print("Error")
                break
        if epoch_total_ones!=0 and epoch_total_zeros!=0 and epoch%acc_epoch==0 and epoch!=0:
            print('Epoch: [%d] | Training_Loss: %.3f | Total_Accuracy: %.3f | Ones_Accuracy: %.3f | Zeros_Accuracy: %.3f |' %
                      (epoch, running_loss / batches_num, 100 * epoch_correct / epoch_total, 100 * \
                          epoch_correct_ones / epoch_total_ones, 100 * epoch_correct_zeros / epoch_total_zeros))
            logging.info('Epoch: [%d] | Training_Loss: %.3f | Total_Accuracy: %.3f | Ones_Accuracy: %.3f | Zeros_Accuracy: %.3f |' %
                      (epoch, running_loss / batches_num, 100 * epoch_correct / epoch_total, 100 * \
                          epoch_correct_ones / epoch_total_ones, 100 * epoch_correct_zeros / epoch_total_zeros))
        else:
            print('Epoch: [%d] | Training_Loss: %.3f |' %
                        (epoch, running_loss / batches_num))
            logging.info('Epoch: [%d] | Training_Loss: %.3f |' %
                        (epoch, running_loss / batches_num))
        # save model
        if epoch%save_model_epochs==0 and epoch!=0:
            torch.save({ 
                        'epoch': epoch,
                        'model_state_dict': complete_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': running_loss,
                        }, './models/epoch_'+str(epoch)+'.pth')

        #validation
        if epoch%validation_epoch==0 and epoch!=0:
            with torch.no_grad():
                epoch_total=0
                epoch_total_ones= 0
                epoch_total_zeros= 0
                epoch_correct=0
                epoch_correct_ones= 0
                epoch_correct_zeros= 0
                running_loss= 0
                batches_num=0
                for batch in testloader:
                    batch_total=0
                    batch_total_ones= 0
                    batch_total_zeros= 0
                    batch_correct= 0
                    batch_correct_ones= 0
                    batch_correct_zeros= 0
                    batches_num+=1
                    output, output2, ground_truth, ground_truth2, det_num, tracklet_num = complete_net(batch)
                    loss = weighted_binary_cross_entropy(output, ground_truth, weights)
                    running_loss += loss.item()
                    ##Accuracy 
                    if epoch%acc_epoch2==0 and epoch!=0:
                        # Hungarian method, clean up
                        cleaned_output= hungarian(output2, ground_truth2, det_num, tracklet_num)
                        batch_total += cleaned_output.size(0)
                        ones= torch.tensor([1 for x in cleaned_output]).to(device)
                        zeros = torch.tensor([0 for x in cleaned_output]).to(device)
                        batch_total_ones += (cleaned_output == ones).sum().item()
                        batch_total_zeros += (cleaned_output == zeros).sum().item()
                        batch_correct += (cleaned_output == ground_truth2).sum().item()
                        temp1 = (cleaned_output == ground_truth2)
                        temp2 = (cleaned_output == ones)
                        batch_correct_ones += (temp1 & temp2).sum().item()
                        temp3 = (cleaned_output == zeros)
                        batch_correct_zeros += (temp1 & temp3).sum().item()
                        epoch_total += batch_total
                        epoch_total_ones += batch_total_ones
                        epoch_total_zeros += batch_total_zeros
                        epoch_correct += batch_correct
                        epoch_correct_ones += batch_correct_ones
                        epoch_correct_zeros += batch_correct_zeros
                if epoch_total_ones!=0 and epoch_total_zeros!=0 and epoch%acc_epoch2==0 and epoch!=0:
                    print('Epoch: [%d] | Validation_Loss: %.3f | Total_Accuracy: %.3f | Ones_Accuracy: %.3f | Zeros_Accuracy: %.3f |' %
                                (epoch, running_loss / batches_num, 100 * epoch_correct / epoch_total, 100 * \
                                    epoch_correct_ones / epoch_total_ones, 100 * epoch_correct_zeros / epoch_total_zeros))
                    logging.info('Epoch: [%d] | Validation_Loss: %.3f | Total_Accuracy: %.3f | Ones_Accuracy: %.3f | Zeros_Accuracy: %.3f |' %
                                (epoch, running_loss / batches_num, 100 * epoch_correct / epoch_total, 100 * \
                                    epoch_correct_ones / epoch_total_ones, 100 * epoch_correct_zeros / epoch_total_zeros))
                else:
                    print('Epoch: [%d] | Validation_Loss: %.3f |' %
                                (epoch, running_loss / batches_num))
                    logging.info('Epoch: [%d] | Validation_Loss: %.3f |' %
                                (epoch, running_loss / batches_num))
    

    
