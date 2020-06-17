from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from glob import glob

import cv2

from detection_classes import mobilenet_detection


def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), (0, 0, 255), 2)


print('Start Recognition!')

face_detection = mobilenet_detection.MobileNetDetection(face_crop_margin=16)
img_list = glob('D:\Documents\GitHub\Face-Detection\images\e*.jp*g')
for img in img_list:
    frame = cv2.imread(img)
    if frame is None:
        continue
    frame = frame[:, :, 0:3]
    faces = face_detection.find_faces(frame)
    
    add_overlays(frame, faces)
    
    cv2.imwrite('Detected/MobileNetSSD/' + os.path.splitext(os.path.basename(img))[0] + '.png', frame)
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        continue

cv2.destroyAllWindows()
