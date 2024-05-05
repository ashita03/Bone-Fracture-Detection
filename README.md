#  Bone Fracture Detection Web Application

This project involves a web application that uses computer vision to predict bone fracture detection. The application utilizes three different models for classification and detection: YOLOv8, Faster R-CNN with ResNet, and VGG16 with SSD.

## Data Source
The dataset used for bone fracture detection is available on Kaggle. You can find the dataset and additional information [here](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project).

## About Data
The dataset includes X-ray images of different bone fractures. The classes available in the dataset are - Elbow Positive, Fingers Positive, Forearm Fracture, Humerus Fracture, Shoulder Fracture, and Wrist Positive. Each image includes a corresponding file with the annotated bounding box of the fracture.

## Running the Repository
To run the web application locally, follow these steps:

* Clone the repository to your local machine.
* Create a virtual environment for the project.
* Install all the required dependencies.
* Download the pre-trained weights for the models [here](https://drive.google.com/drive/folders/15LnW-DVp9VOx7-hPbCGKrfb8Ot0m_qlA?usp=sharing). Add them to a 'weights' folder in the project directory.
* Run the following command in your terminal to run the app
  ```
  streamlit run app.py
  ```

## Web Application Features
The web application is built using Streamlit, providing user-friendly options for bone fracture detection:

### Left Panel
* Model Selection: Users can choose from the three available models: YOLOv8, Faster R-CNN with ResNet, and VGG16 with SSD.
* Confidence Threshold: A slider in the left panel allows users to set the prediction confidence threshold.

### Overview Tab
The Overview tab provides information about the different models used for bone fracture detection, their significance, and performance metrics.

### Testing Tab
In the Testing tab, users can test the models by uploading images and observing the detection results that includes the output class and the respective bounding box and scores.


##  Working Demo

https://github.com/ashita03/Bone-Fracture-Detection/assets/66629830/00368de2-d025-499e-9369-3d4f4d981098


## Get Started
To get started with the bone fracture detection web application, you'll need to clone this repository and follow the instructions above to run the application locally. Experiment with different models and confidence thresholds to explore their performance.

For any issues or feedback, please feel free to reach out

Enjoy detecting bone fractures with computer vision!
