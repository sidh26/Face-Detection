import time

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import label_map_util


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class MobileNetDetection:
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = 'models\\frozen_inference_graph_face.pb'
    
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'protos\\face_label_map.pbtxt'
    
    NUM_CLASSES = 2
    
    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.detection_graph, self.detection_graph_sess = self._setup_mobilenet()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
    
    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
    
    def find_faces(self, image):
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.detection_graph_sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        # print('inference time cost: {}'.format(boxes.shape))
        faces = []
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        
        for i in range(boxes.shape[0]):
            if scores[i] > 0.7:
                face = Face()
                face.container_image = image
                face.bounding_box = np.zeros(4, dtype=np.int32)
                box = tuple(boxes[i].tolist())
                ymin, xmin, ymax, xmax = box
                im_width, im_height = np.asarray(image.shape)[0:2][1], np.asarray(image.shape)[0:2][0]
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                img_size = np.asarray(image.shape)[0:2]
                face.bounding_box[0] = np.maximum(left - self.face_crop_margin / 2, 0)
                face.bounding_box[1] = np.maximum(top - self.face_crop_margin / 2, 0)
                face.bounding_box[2] = np.minimum(right + self.face_crop_margin / 2, img_size[1])
                face.bounding_box[3] = np.minimum(bottom + self.face_crop_margin / 2, img_size[0])
                cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
                face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
                faces.append(face)
        return faces
    
    def _setup_mobilenet(self, ):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # with detection_graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return detection_graph, tf.Session(graph=detection_graph, config=config)
