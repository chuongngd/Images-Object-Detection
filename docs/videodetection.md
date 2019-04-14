## Overwiew
User initialize live stream video through the web application. The application uses computer's camera to test the application. 
The computer's camera can replace by any ip camera that can connect to the localhost.
The Flask server will call the Object Detection API to detect the video frame and return the result as an image.
The Object Detection API will extract object name, location, the accuracy of the objects in the video frame and return JSON data type.
The video frame after detection will be redrawn and all of the objects in the video frame will be marked by a bounding box with object name and accuracy percentage.
## This function describe how to capture the video frame and detect each frame, then display back on the web browser

def detect_video():<br/>
    ret = True <br/>
    while(ret): <br/>
        cap = cv2.VideoCapture(0) <br/>
        ret,image_np = cap.read() <br/>
        output_dict = run_inference_for_single_image(image_np, detection_graph) <br/>
        vis_util.visualize_boxes_and_labels_on_image_array(image_np,output_dict['detection_boxes'],output_dict['detection_classes'],
                                                            output_dict['detection_scores'],category_index,
                                                            instance_masks=output_dict.get('detection_masks'),
                                                            use_normalized_coordinates=True,line_thickness=8) <br/>
        ret,jpeg = cv2.imencode('.jpg',image_np) <br/>
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n') <br/>
        if cv2.waitKey(1) & 0xFF == ord('q'): <br/>
            cv2.destroyAllWindows() <br/>
            cap.release() <br/>
            break 
### Object Detection API can be found [here](https://github.com/tensorflow/models/tree/master/research/object_detection)
## Sequence Diagram
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/sequence%20detect%20vide.png)
## Flowchart
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/detectvideoflow.png)
## Example of a detected video
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/videocapture_book21-11-55.jpg)
