# USAGE
# python intersection_over_union.py

import os
from glob import glob

# import the necessary packages
import cv2
import pandas as pd


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou


annotations = pd.read_csv('ground_truth_jpg_51.csv')

mobilenet = pd.read_csv('mobilenet_detection.csv')

mtcnn = pd.read_csv('mtcnn_detection.csv')
mtcnn[['C1', 'C2', 'C3', 'C4']] = mtcnn[['C1', 'C2', 'C3', 'C4']].astype('int')

resnet = pd.read_csv('resnet_detection.csv')
resnet[['C1', 'C2', 'C3', 'C4']] = resnet[['C1', 'C2', 'C3', 'C4']].astype('int')

maskrcnn = pd.read_csv('maskrcnn_detection.csv')
maskrcnn[['C1', 'C2', 'C3', 'C4']] = maskrcnn[['C1', 'C2', 'C3', 'C4']].astype('int')

mnious = []
mtious = []
rnious = []
mrious = []

for path in glob('images/cctv/*.jp*g'):
    filename = os.path.basename(path)
    print(filename)
    
    c1, c2, c3, c4 = annotations[annotations['Image Name'] == filename][['C1', 'C2', 'C3', 'C4']].values[0]
    c3 += c1
    c4 += c2
    gt = [c1, c2, c3, c4]
    
    mnpred = list(mobilenet[mobilenet['Image Name'] == filename][['C1', 'C2', 'C3', 'C4']].values[0])
    mniou = bb_intersection_over_union(gt, mnpred)
    mnious.append(mniou)
    
    mtpred = list(mtcnn[mtcnn['Image Name'] == filename][['C1', 'C2', 'C3', 'C4']].values[0])
    mtiou = bb_intersection_over_union(gt, mtpred)
    mtious.append(mtiou)
    
    rnpred = list(resnet[resnet['Image Name'] == filename][['C1', 'C2', 'C3', 'C4']].values[0])
    rniou = bb_intersection_over_union(gt, rnpred)
    rnious.append(rniou)
    
    # mrpred = list(maskrcnn[maskrcnn['Image Name'] == filename][['C1', 'C2', 'C3', 'C4']].values[0])
    # mriou = bb_intersection_over_union(gt, mrpred)
    # mrious.append(mriou)
    
    img = cv2.imread(path)
    cv2.rectangle(img, tuple(gt[:2]), tuple(gt[2:]), (0, 255, 0), 5)
    
    img1 = img.copy()
    cv2.rectangle(img1, tuple(mnpred[:2]), tuple(mnpred[2:]), (0, 0, 255), 5)
    img1 = cv2.resize(img1, (0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
    cv2.putText(img1, "IoU: {:.4f}".format(mniou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    img2 = img.copy()
    cv2.rectangle(img2, tuple(mtpred[:2]), tuple(mtpred[2:]), (0, 0, 255), 5)
    img2 = cv2.resize(img2, (0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
    cv2.putText(img2, "IoU: {:.4f}".format(mtiou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    img3 = img.copy()
    cv2.rectangle(img3, tuple(rnpred[:2]), tuple(rnpred[2:]), (0, 0, 255), 5)
    img3 = cv2.resize(img3, (0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
    cv2.putText(img3, "IoU: {:.4f}".format(rniou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # img4 = img.copy()
    # cv2.rectangle(img4, tuple(mrpred[:2]), tuple(mrpred[2:]), (0, 0, 255), 5)
    # img4 = cv2.resize(img4, (0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
    # cv2.putText(img4, "IoU: {:.4f}".format(mriou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # print("{}: {:.4f}".format(filename, iou))
    
    cv2.imshow('MobileNet', img1)
    cv2.imshow('MTCNN', img2)
    cv2.imshow('ResNet', img3)
    # cv2.imshow('MaskRCNN', img4)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        continue
cv2.destroyAllWindows()

sum(mnious) / len(mnious)
sum(mtious) / len(mtious)
sum(rnious) / len(rnious)
sum(mrious) / len(mrious)

annotations = pd.read_csv('ground_truth_jpg_51.csv')


def iou(pred_path):
    detected_ious = []
    all_ious = []
    pred = pd.read_csv(pred_path)
    pred[['C1', 'C2', 'C3', 'C4']] = pred[['C1', 'C2', 'C3', 'C4']].astype('int')
    unpred = 0
    for index, value in pred.iterrows():
        pred_bb = value[['C1', 'C2', 'C3', 'C4']].tolist()
        c1, c2, c3, c4 = annotations[annotations['Image Name'] == value['Image Name']][['C1', 'C2', 'C3', 'C4']].values[0]
        c3 += c1
        c4 += c2
        gt = [c1, c2, c3, c4]
        all_ious.append(bb_intersection_over_union(gt, pred_bb))
        if pred_bb == [0, 0, 0, 0]:
            unpred += 1
            continue
        detected_ious.append(bb_intersection_over_union(gt, pred_bb))
    return sum(all_ious) / len(all_ious), sum(detected_ious) / len(detected_ious), unpred


iou('mobilenet_detection_102.csv')
iou('mobilenet_detection_51.csv')
iou('mtcnn_detection_102.csv')
iou('mtcnn_detection_51.csv')
iou('resnet_detection_102.csv')
iou('resnet_detection_51.csv')
iou('maskrcnn_detection_102.csv')
iou('maskrcnn_detection_51.csv')
