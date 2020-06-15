# import the necessary packages

import os
from glob import glob

import cv2
import numpy as np

# construct the argument parse and parse the arguments

# load our serialized model from disk
print("[INFO] loading model...")

net = cv2.dnn.readNetFromCaffe(r'ResNetSSD\deploy.txt', r'ResNetSSD\model.caffemodel')
print("[INFO] Model LOADED")
img_list = glob('D:\Documents\GitHub\Face-Detection\images\e*.jpg')
for img in img_list:
    
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    
    image = cv2.imread(img)
    (h, w) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (np.mean(image[:, :, 2]), np.mean(image[:, :, 1]), np.mean(image[:, :, 0])), swapRB=True)
    
    # takes care of pre-processing which includes
    # setting the blob  dimensions and normalization.
    # blob is essentially the same image with the same spatial dimensions (height and width), same depth (channels)
    # that have been preprocessed in some manner
    # we have a scale factor of 1.0
    
    # pass the blob through the network and obtain the detections and
    # predictions
    
    print("[INFO] computing face detections...")
    
    net.setInput(blob)
    detections = net.forward()
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        
        # extract the confidence (i.e., probability) associated with the
        # prediction
        
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        
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
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            # cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        # show the output image
    cv2.imwrite('Detected/ResNetSSD/' + os.path.splitext(os.path.basename(img))[0] + '.png', image)
    cv2.imshow("Output", image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        continue
cv2.destroyAllWindows()
