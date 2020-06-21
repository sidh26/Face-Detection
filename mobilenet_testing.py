from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from glob import glob
from time import time

import cv2
import pandas as pd

from detection_classes import mobilenet_detection


def add_overlays(frame, faces):
    bounding_box = 0, 0, 0, 0
    if faces is not None:
        max_score_ind = 0
        max_score = 0
        for i in range(len(faces)):
            if faces[i].score > max_score:
                max_score = faces[i].score
                max_score_ind = i
        face = faces[max_score_ind]
        face_bb = face.bounding_box.astype(int)
        cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), (0, 0, 255), 2)
        bounding_box = face_bb[0], face_bb[1], face_bb[2], face_bb[3]
    return bounding_box


print('Start Recognition!')
rows = []
face_detection = mobilenet_detection.MobileNetDetection(face_crop_margin=16)
img_list = glob('D:\Documents\GitHub\Face-Detection\images\class\*.jpg')
total_time = 0
for img in img_list:
    frame = cv2.imread(img)
    if frame is None:
        continue
    frame = frame[:, :, 0:3]
    start_time = time()
    faces = face_detection.find_faces(frame)
    end_time = time()
    total_time += end_time - start_time
    bounding_box = add_overlays(frame, faces)
    rows.append([os.path.basename(img), *bounding_box])
    cv2.imwrite('Detected/MobileNetSSD/' + os.path.splitext(os.path.basename(img))[0] + '.png', frame)
    # frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    # cv2.imshow('Video', frame)
    #
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     continue
df = pd.DataFrame(rows, columns=['Image Name', 'C1', 'C2', 'C3', 'C4'])
df.to_csv('mobilenet_detection.csv')
cv2.destroyAllWindows()
