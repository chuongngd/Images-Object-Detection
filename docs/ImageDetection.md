## Overwiew
User inputs the image through the web application. 
The Flask server will call the Object Detection API to detect the image and return the list of objects in the image
The Object Detection API will extract object name, location, the accuracy of the objects in the image and return JSON data type.
The images after detection will be redrawn and all of the objects in the image will be marked by a bounding box with object name and accuracy percentage.
## Example of JSON data type of an image
{
      "type" : "object",
      "required" : [ "height", "id", "name", "score", "width", "x", "y" ],
      "properties" : [
        {
			"score" : 0.8791,
			"name" : "person",
			"width" : 0.5467,
			"x" : 0.0192,
			"y" : 0.3956,
			"id" : 1.0,
			"height" : 0.9536
		},
		{
			"score" : 0.7491,
			"name" : "dog",
			"width" : 0.3467,
			"x" : 0.1292,
			"y" : 0.5656,
			"id" : 2.0,
			"height" : 0.5636
		}
    ]
  }
## Sequence Diagram
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/sequencedetectimage.png)
## Flowchart
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/detectimageflow.png)
## Example of a detected image
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/green-pearlescent-wrap-san-diego.jpeg)
