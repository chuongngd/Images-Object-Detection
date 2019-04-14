## General Techical Approach
Create a web application which user can use as an online tool to detect an object from images. Image after detection will return a list of objects.  The application will be designed to leverage controller and view in a Flask framework. A Python application uses a TensorFlow Object Detection model to provides an API to the web application to detect images’ objects. A MySQL database is used store object detection results. 
## Key Technical Design Decisions:
- Embedding: The application is required TensorFlow Object Detection API. A Python application using TensorFlow will be running on the server to create a TensorFlow Object Detection API. This application needed a high-performance machine to perform training and testing Object Detection model. In this project scope, a computer with GPU is used to host the Flask server and run the TensorFlow Object Detection API. <br/>
- Back End Service: The TensorFlow Object Detection application is required to process and detect objects in the image; a back-end service using Flask will consume TensorFlow Object Detection API and get the detail of the objects to belong to the image to store in the database. The MySQL database is used for the project. <br/>
- Front End Web Application: A front-end web application using HTML, JavaScript to implement a detect image’s object application. The application will leverage some common functions and display the object detection result, as well as display the images which contain specific objects when user look for it. <br/>
- TensorFlow Object Detection API: is an open source framework built on top of TensorFlow that makes it easy to construct, train and deploy object detection models. TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) . In this project, a MobileNet SSD model is used to train the dataset and export the trained model that can use to detect the image’s object.
## Proof of Concepts
### TensorFlow
TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networkt
### PASCAL VOC 2012
To provide training set of labelled images. There are twenty object classes that have been selected in the dataset are:
> Person: person <br/>
> Animal: bird, cat, cow, dog, horse, sheep <br/>
> Vehicle: aero plane, bicycle, boat, bus, car, motorbike, train <br/>
> Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
### MobileNet SSD (Single Shot Detection)
The algorithm to detect object from image
### Flask
To run the application. In this project, Flask server is deployed on Azure Virtual Machine
### MySQL database
To store user information, Image and it’s object detection
## Hardware and Software Technologies
### MySQL 5.6: open-source relational database management system. It I GPL(version 2) license 
### Python 3.6: developed by Python Software Foundation
### Flask 1.0.2: microframework for Python. It is BSD licensed.
### TensorFlow 1.10:  
open source software library for high performance numerical computation. Originally developed by researchers and engineers from the Google Brain team within Google’s AI organization
### PASCAL VOC 2012 dataset from PASCAL VOC project. 
### Windows 10 2.2 GHZ x 16GB RAM x 8GBGPU x 250 GB SSD
