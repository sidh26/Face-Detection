# import the necessary packages

import os
from glob import glob
from time import time

import cv2
import numpy as np
import pandas as pd

# construct the argument parse and parse the arguments

# load our serialized model from disk
print("[INFO] loading model...")

net = cv2.dnn.readNetFromCaffe(r'ResNetSSD\deploy.txt', r'ResNetSSD\model.caffemodel')
print("[INFO] Model LOADED")

img_list = glob('D:\Documents\GitHub\Face-Detection\images\cctv\*.jp*g')
# img_list.extend(img_list)

rows = []
total_time = 0
for img in img_list:
    
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    
    frame = cv2.imread(img)
    (h, w) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (np.mean(frame[:, :, 2]), np.mean(frame[:, :, 1]), np.mean(frame[:, :, 0])), swapRB=True)
    
    # takes care of pre-processing which includes
    # setting the blob  dimensions and normalization.
    # blob is essentially the same image with the same spatial dimensions (height and width), same depth (channels)
    # that have been preprocessed in some manner
    # we have a scale factor of 1.0
    
    # pass the blob through the network and obtain the detections and
    # predictions
    
    print("[INFO] computing face detections...")
    start_time = time()
    net.setInput(blob)
    detections = net.forward()
    end_time = time()
    total_time += end_time - start_time
    scores = detections[0, 0, :, 2]
    if len(scores) > 0 and max(scores) > 0.25:
        bounding_box = detections[0, 0, np.argmax(scores), 3:7] * np.array([w, h, w, h])
        temp = frame[int(bounding_box[1]): int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2]), :]
        cv2.imwrite('Detected/MTCNN/cropped51/' + os.path.splitext(os.path.basename(img))[0] + '.png', temp)
    else:
        bounding_box = np.array([0, 0, 0, 0])
    rows.append([os.path.basename(img), *bounding_box])
    # loop over the detections
    for i in range(0, detections.shape[2]):
        
        # extract the confidence (i.e., probability) associated with the prediction
        
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        
        if confidence > 0.25:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            # Drawing rectangle and % possibility
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            # cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        # show the output image
    cv2.imwrite('Detected/ResNetSSD/' + os.path.splitext(os.path.basename(img))[0] + '.png', frame)
    # image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    # cv2.imshow("Output", image)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     continue

total_time / len(img_list)
df = pd.DataFrame(rows, columns=['Image Name', 'C1', 'C2', 'C3', 'C4'])
df.to_csv('resnet_detection_51.csv')
cv2.destroyAllWindows()
