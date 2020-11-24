from data_preparation_train import *
from model_training import *
from model_testing import *
from network.complete_net import *
from utils import *
import os
import pickle
import argparse
import pprint

def get_data(info, set):
    if set=='training':
        detections = np.loadtxt("/data/MOT17/train/MOT17-{}-{}/gt/gt.txt".format(info[0], info[1]), delimiter=',')
        images_path = "/data/MOT17/train/MOT17-{}-{}/img1".format(info[0], info[1])
    elif set=='testing':
        detections = np.loadtxt("/data/MOT17/test/MOT17-{}-{}/det/det.txt".format(info[0], info[1]), delimiter=',')
        images_path = "/data/MOT17/test/MOT17-{}-{}/img1".format(info[0], info[1])
    return detections, images_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str)
    args = parser.parse_args()

    if args.type == 'train':

        # initialize training settings
        batchsize = 2 # specify how many graphs to use at each batch
        epochs = 4 # at how many epochs to stop
        load_checkpoint= None#'./models/epoch_2.pth' # None or specify .pth file to continue training
        validation_epochs= 4 # epoch interval for validation
        acc_epoch = epochs # at which epoch to calculate training accuracy
        acc_epoch2 = epochs # at which epoch to calculate validation accuracy
        save_model_epochs = 1 # epoch interval to save the model
        most_recent_frame_back = 30 # for more challenging training, specify most recent frame of tracklets to match new detections with, a max value is specified here 
                                    # so later it will be randomly between 1 and most_recent_frame_back, min value=1 -> take only previous frame
        frames_look_back = 30  # a second limit that is used to use tracklets for matching in frames between frames_look_back and most_recent_frame_back,
                                # min value=1 -> take only previous frame
        # example: if current frame= 60, tracklets used randomly from t1= 30 to 59, also tracklets used till t1-30
        graph_jump = 5  # how many frames to move the current frame forward, min value=1 -> move to the next frame
        distance_limit = 250 # objects within that pixel distance can be associated 

        # MOT17 specific settings
        train_seq = ["02", "04", "05", "09", "10", "11", "13"] # names of videos
        valid_seq = ["02", "04", "05", "09", "10", "11", "13"] # names of videos
        fps= [30,30,14,30,30,30,25] # specify fps of each video
        current_frame_train = 2 # use as first current frame the second frame of each video
        total_frames = [None, None, None, None, None, None, None] # total frames of each video loaded, None for max frames
        current_frame_valid = [500,900,780,450,550,800,650] # up to which frame of each video to use for training
        detector = ["FRCNN"] # specify a detector just to direct to one of the MOT folders

        # Option 1: If graph data not built, loop through sequence and get training data
        print('\n')
        print('Training Data')
        data_list_train = []
        for s in range(len(train_seq)):
            for d in range(len(detector)):
                print('Sequence: ' + train_seq[s])
                detections, images_path = get_data([train_seq[s], detector[d]], "training")
                list = data_prep_train(train_seq[s], detections, images_path, frames_look_back, total_frames[s], most_recent_frame_back,
                                        graph_jump, current_frame_train, current_frame_valid[s], distance_limit, fps[s], "training")
                data_list_train = data_list_train + list
        with open('./data/data_train.data', 'wb') as filehandle:
            pickle.dump(data_list_train, filehandle)
        print("Saved to pickle file \n")
        print('Validation Data')
        data_list_valid = []
        for s in range(len(valid_seq)):
            for d in range(len(detector)):
                print('Sequence: ' + valid_seq[s])
                detections, images_path = get_data([valid_seq[s], detector[d]], "training")
                list = data_prep_train(valid_seq[s], detections, images_path, frames_look_back, total_frames[s], most_recent_frame_back,
                                        graph_jump, current_frame_train, current_frame_valid[s], distance_limit, fps[s], "validation")
                data_list_valid = data_list_valid + list
        with open('./data/data_valid.data', 'wb') as filehandle:
            pickle.dump(data_list_valid, filehandle)
        print("Saved to pickle file \n")

        # Option 2: If data graph built, just import files
        # with open('./data/data_train.data', 'rb') as filehandle:
        #     data_list_train = pickle.load(filehandle)
        # print("Loaded training pickle files")
        # with open('./data/data_valid.data', 'rb') as filehandle:
        #     data_list_valid = pickle.load(filehandle)
        # print("Loaded validation pickle files")
        #Load and train
        model_training(data_list_train, data_list_valid, epochs, acc_epoch, acc_epoch2, save_model_epochs, validation_epochs, batchsize, "logfile", load_checkpoint)

    elif args.type == 'test':

        frames_look_back = 30 # how many previous frames to look back for tracklets to be matched
        match_thres= 0.25 # the matching confidence threshold
        det_conf_thres= 0.0 # the lowest detection confidence
        distance_limit = 200 # objects within that pixel distance can be associated 
        min_height= 10 #minimum height of detections
        fp_look_back= 15 # for false positives
        fp_recent_frame_limit= 10 # for false positives
        fp_min_times_seen= 3 # for false positives, minimum times seen for the last fp_look_back frames 
                            #with the first instance seen before fp_recent_frame_limit otherwise considered false positive 

        # Select sequence
        seq = ["01","03","06","07","08","12","14"]
        fps= [30,30,14,30,30,30,25]
        # detector = ["DPM", "FRCNN", "SDP"]
        detector = ["FRCNN"]

        #load model
        model = completeNet()
        device = torch.device('cuda')
        model = model.to(device)
        model = DataParallel(model)
        model.load_state_dict(torch.load('./models/epoch_11.pth')['model_state_dict'])
        model.eval()

        # Load data and test, write output to txt and video
        data_list = []
        for s in range(len(seq)):
            for d in range(len(detector)):
                detections, images_path = get_data([seq[s], detector[d]], "testing")
                total_frames= None # None for max frames
                if total_frames == None:
                    total_frames = np.max(detections[:, 0])  # change only if you want a subset of the total frames
                print('Sequence: '+ seq[s])
                print('Total frames: ' + str(total_frames))
                detections = sorted(detections, key=lambda x: x[0])
                tracking_output= model_testing(seq[s], detections, images_path, total_frames, frames_look_back, model, distance_limit, fp_min_times_seen, match_thres, det_conf_thres, fp_look_back, fp_recent_frame_limit, min_height, fps[s])
                #write output
                outputFile = './output/MOT17-{}-{}.avi'.format(seq[s],detector[d])
                #get dimensions of images
                image_name = os.path.join(images_path, "{0:0=6d}".format(1) + ".jpg")
                image = cv2.imread(image_name, cv2.IMREAD_COLOR)
                vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                             (image.shape[1], image.shape[0]))
                with open('./output/MOT17-{}-{}.txt'.format(seq[s],detector[d]), 'w') as f:
                    for frame in range(1,int(total_frames)+1):
                        image_name = os.path.join(images_path, "{0:0=6d}".format(int(frame)) + ".jpg")
                        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
                        for item in tracking_output:
                            if item[0]==frame:
                                #write tracking output to txt
                                f.write('%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n' % (item[0], item[1], item[2], item[3], item[4], item[5], -1, -1, -1, -1))
                                #write tracking output to frame
                                xmin = int(item[2])
                                ymin = int(item[3])
                                xmax = int(item[2] + item[4])
                                ymax = int(item[3] + item[5])
                                display_text = '%d' % (item[1])
                                color_rectangle = (0, 0, 255)
                                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color_rectangle, thickness=2)
                                font = cv2.FONT_HERSHEY_PLAIN
                                color_text = (255, 255, 255)
                                cv2.putText(image, display_text,
                                            (xmin + int((xmax - xmin) / 2), ymin + int((ymax - ymin) / 2)), fontFace=font,
                                            fontScale=1.3, color=color_text,
                                            thickness=2)
                            elif item[0]>frame:
                                break
                        for item in detections:
                            if item[0] == frame and item[2]>0 and item[3]>0 and item[4]>0 and item[5]>0:
                                xmin = int(item[2])
                                ymin = int(item[3])
                                xmax = int(item[2] + item[4])
                                ymax = int(item[3] + item[5])
                                color_rectangle = (255, 255, 255)
                                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color_rectangle, thickness=1)
                            elif item[0]>frame:
                                break
                        xmin = 0
                        ymin = 0
                        xmax = 1920
                        ymax = 1080
                        display_text = 'Frame %d' % (frame)
                        font = cv2.FONT_HERSHEY_PLAIN
                        color_text = (0, 0, 255)
                        cv2.putText(image, display_text,
                                    (50, 50), fontFace=font,
                                    fontScale=1.3, color=color_text,
                                    thickness=2)
                        vid_writer.write(image)
                cv2.destroyAllWindows()
                vid_writer.release()

















