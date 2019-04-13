# Image's Object Detection
## Abstract
The application uses TensorFlow Object Detection API and Flask Python to create an application for image detection. User input image and the object detection algorithm will return the objects in the image. All the images and objects will be saved into the database. The application also detect objects from stream video, and capture the stream video with its object.

## Video Overview
https://www.loom.com/share/b38373419963477faaa9f4d3b8eb0335

## Funtionals Requirements
The application will implement user registration module, login module, and object detection module. 
The application separate presentation layer, business layer and database layer. 
The application separates the view (jinja2 templates) and the controllers. 
The application uses TensorFlow Object Detection API to detect image’s objects.
The application use relational database MySQL. 
The application is deployed on Flask server and localhost. 


## Logical Solution Design
The solution divides the application to Controller (Python site) and View (Jinja2 templates). The server side includes a Flask server and TensorFlow Object Detection API.

![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/logical.png)
## Physical Architecture
The Flask server, TensorFlow Object Detection API, and MySQL database are hosted and demo on localhost (Windows OS CPU 2.2GHZ x 8GB GPU x 16 GB RAM x 16 GB Storage). 
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/physical.png)

## Component Design
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/component.png)

## Deployment Diagram
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/deployment.png)

## Database
A high-level view of the database, showing how the tables relate to each other and what rows there are in each table.
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/ER.png)

## Sitemap
An overview of the flow of the website.
![alt tag](https://github.com/chuongngd/Images-Object-Detection/blob/master/pictures/sitemap.png)

## Technical Design
A brief overview of the general approach and the main libraries that were leveraged can be found here
## Funtionalities 



### Installing



## Running the tests

Explain how to run the automated tests for this system


## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Citations

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

