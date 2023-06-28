import os
import numpy as np
import tensorflow as tf
import cv2 # For real-time detection
import argparse
from PIL import Image # For static detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

def real_time_detection():
    PATH_TO_MODEL_DIR = "./exported-models/my_model"
    PATH_TO_LABELS = "./annotations/label_map.pbtxt"
    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    
    # define a video capture object
    vid = cv2.VideoCapture(0)
    
    while(True):
        # Capture the video frame
        ret, frame = vid.read()
    
        # Display the resulting frame
        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        image_with_detections = frame.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0.5, # threshold
            agnostic_mode=False)
        cv2.imshow("frame", image_with_detections)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    cv2.destroyAllWindows()

def static_detection():    
    PATH_TO_DATA = "./data/images"
    PATH_TO_SAVE = "./data/output"
    PATH_TO_MODEL_DIR = "./exported-models/my_model"
    PATH_TO_LABELS = "./annotations/label_map.pbtxt"
    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    for image_path in os.listdir(PATH_TO_DATA):
        image = Image.open(os.path.join(PATH_TO_DATA, image_path))
        image = np.asarray(image)
    
        # Display the resulting image
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        image_with_detections = image.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0.5, # threshold
            agnostic_mode=False)
        
        image_detected = Image.fromarray(image_with_detections)
        image_detected.save(os.path.join(PATH_TO_SAVE, image_path))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, help="Select the mode of real-time/static")

    args = parser.parse_args()

    mode = args.mode

    if mode == "real-time":
        real_time_detection()
    if mode == "static":
        static_detection()

if __name__ == "__main__":
    main()