# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 23:14:16 2019

@author: chuon
"""
import numpy as np
import glob, os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib import cm
from dbconnect import connection
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import ops as utils_ops
#from object_detection import core
from utils import label_map_util

from utils import visualization_utils as vis_util

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# Model Preparation

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PHOTOS_FOLDER = os.path.join('images/')
PATH_TO_LABELS = 'C:/Users/chuon/application1/env/app/object_detection/object_detection/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90

# Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# added to put object in JSON
class Object(object):
    def __init__(self):
        self.name=" TensorFlow Object Detection REST API"

#create JSON attribute of object
    def toJSON(self,name, class_name, score,y,x,height,width):
        result = {}
        result["name"] = name
        result["class_name"] = class_name
        result["score"] = score
        result["y"] = y
        result["x"] = x
        result["height"] = height
        result["width"] = width
        return result
#function get objects from the image, it returns json list of objects
def get_objects(image):
    threshold=0.5
    sess = tf.Session(graph=detection_graph)
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection. The tensorflow api runs the detection 
    # It return the value of objects(classes), scores(accuracy), boxes (location), num(number of object) 
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    
    #convert the classes, scores and boxes dimession to numpy array
    classes = np.squeeze(classes).astype(np.int64)
    scores = np.squeeze(scores)
    boxes = np.squeeze(boxes)

    obj_above_thresh = sum(n > threshold for n in scores)
    

    # Add another json data field to the output
    item = {}
    item["version"] = "0.0.1"
    item["numObjects"] = obj_above_thresh
    item["threshold"] = threshold
    outputJson = []
    outputJson.append(item)
    #convert detection result to json format 
    big_json={}
    big_json["array"] = outputJson
    #put object from actual detection result into json output
    for c in range(0, len(classes)):
        class_name = category_index[classes[c]]['name']
        if scores[c] >= threshold:      # only return confidences equal or greater than the threshold
            item = Object()
            item.name = 'Object'
            item.class_name = class_name
            item.score = float(scores[c])
            item.y = float(boxes[c][0])
            item.x = float(boxes[c][1])
            item.height = float(boxes[c][2])
            item.width = float(boxes[c][3])
            itemJson = item.toJSON(item.name, item.class_name, item.score,
                                   float(scores[c]),float(boxes[c][0]),
                                   float(boxes[c][2]),float(boxes[c][3]))
            outputJson.append(itemJson)
            
            big_json["array"] = outputJson
      
    sess.close()
    return big_json

#function detect image and draw object box
def detect_image_draw(image,filename):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image_np)
  output_dir = 'C:/Users/chuon/application1/env/app/images/'
  #save the image into local host
  plt.savefig(output_dir + filename)

  
import cv2
 
#function to detect a video , it get frame by read each video capture, detect objects on the video capture
# then display the frame by frame to create the video
def detect_video():
    ret = True
    #cap = cv2.VideoCapture(0)
    while(ret):
        cap = cv2.VideoCapture(0)
        ret,image_np = cap.read()
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(image_np,output_dict['detection_boxes'],output_dict['detection_classes'],
                                                           output_dict['detection_scores'],category_index,
                                                           instance_masks=output_dict.get('detection_masks'),
                                                           use_normalized_coordinates=True,line_thickness=8)
        ret,jpeg = cv2.imencode('.jpg',image_np)
        #frame = get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break


