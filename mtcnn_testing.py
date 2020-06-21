from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from glob import glob
from time import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from detection_classes import mtcnn_detection

print('Creating networks and loading parameters')
total_time = 0
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = mtcnn_detection.create_mtcnn(sess, './models/')
        
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        frame_interval = 3
        img_list = glob('D:\Documents\GitHub\Face-Detection\images\class\*.jpg')
        
        print('Start Recognition!')
        rows = []
        for img in img_list:
            frame = cv2.imread(img)
            if frame is None:
                continue
            # frame = cv2.resize(frame, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)
            frame = frame[:, :, 0:3]
            
            ## Use MTCNN to get the bounding boxes
            start_time = time()
            bounding_boxes, _ = mtcnn_detection.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            end_time = time()
            total_time += end_time - start_time
            bounding_box = bounding_boxes[0][0:4]
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]
                
                bb = np.zeros((nrof_faces, 4), dtype=np.int32)
                
                for i in range(nrof_faces):
                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]
                    
                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                        print('face is inner of range!')
                        continue
                    
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)
            else:
                print('Unable to align')
            rows.append([os.path.basename(img), *bounding_box])
            print(os.path.basename(img))
            cv2.imwrite('Detected/MTCNN/' + os.path.splitext(os.path.basename(img))[0] + '.png', frame)
            # frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
            # cv2.imshow('Video', frame)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     continue
        df = pd.DataFrame(rows, columns=['Image Name', 'C1', 'C2', 'C3', 'C4'])
        df.to_csv('mtcnn_detection.csv')
        cv2.destroyAllWindows()
