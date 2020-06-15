from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from glob import glob

import cv2

from detection_classes import mobilenet_detection


def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), (0, 255, 0), 2)
            if face.name is not None:
                if face.name == 'Unknown':
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2, lineType=2)
    
    # cv2.putText(frame, str(frame_rate) + " fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)


print('Creating networks and loading parameters')

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
frame_interval = 3

# video_capture = cv2.VideoCapture(0)
c = 0

print('Start Recognition!')
prevTime = 0
counter = 0
face_detection = mobilenet_detection.MobileNetDetection()
img_list = glob('D:\Documents\GitHub\Face-Detection\images\e*.jpg')
for img in img_list:
    frame = cv2.imread(img)
    # while True:
    #     ret, frame = video_capture.read()
    curTime = time.time()  # calc fps
    timeF = frame_interval
    if frame is None:
        break
    if c % timeF == 0:
        # frame = cv2.resize(frame, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)
        frame = frame[:, :, 0:3]
        faces = face_detection.find_faces(frame)
        
        add_overlays(frame, faces)
    
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / sec
    str = 'FPS: %2.3f' % fps
    text_fps_x = len(frame[0]) - 150
    text_fps_y = 20
    # cv2.putText(frame, str, (text_fps_x, text_fps_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
    # c += 1
    cv2.imwrite('Detected/MobileNetSSD/' + os.path.splitext(os.path.basename(img))[0] + '.png', frame)
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        continue

# video_capture.release()
cv2.destroyAllWindows()
print(counter)
