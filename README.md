# AI-Road-Accident-Detection

Automatic incident detection on Indian Roads using Artificial Intelligence. Model is trained using Tensorflow object detection API.

## Why is it importatnt?

According to report by Ministry of Road Transport and Highways (Transport Research Wing), in 2016 a total of 4,80,652 accidents took place on Indian Roads which resulted into 1,50,785 deaths (31.37%).

Aim of this research work is to reduce this ratio of deaths from 31.37% to 0%. If this system is integrated with an emergency service such as Ambulance, death rates can be reduced by reaching to accident spot immediately.

# Two demos

Use of this AI model in real world is demonstrated using two different demo applications. This proves that this AI model can be integrated with any existing application and provide economical way to detect road accidents in real time and help prevent 

## 1. Python Demo

**Python Demo** directory contains the complete Python demo. Steps for executing are as follows:

1. Go to Python Demo folder and install all the requirements by typing following command:
*pip install -r requirements.txt*
2. Run *classifier.py* using following syntax:

*python classifier.py [Input Image] [Output Image]*

Example, *python classifier.py input.jpg output.jpg*

3. Output image will be created in same directory with name specified in command and same size as input image but labelled boxes containing probability of car accident in image.

**NOTE:** This model is exclusively trained on road conditions in India. Please keep this in mind while giving input image.

**Input Image**

![Input Image](input.jpg?raw=true "Input Image")

**Output Image**

![Output Image](output.jpg?raw=true "Output Image")

## 2. Android Demo

**Android App** directory contains source code of Android app that uses same model to detect car accident on image captured from mobile camera.

There are two ways to interact with Android demo:

1. In case you want to test this app immediately, use the APK file already build and available in *Android App/app/build/outputs/* directly. This APK will be installed on any Android device running on API level 16 or higher.
2. Build the application on your own and test in emulator or your device. This method is suitable if you want to play around with source code (like changing threshold value which is currently **60%**).

**Mobile App Demo Screenshot**

![Android App Demo Screenshot](screenshot.PNG?raw=true "Android App Demo Screenshot")

# LICENSE

This project follows GNU Affero GPL v3.0. Please go through LICENSE before using or distributing this application. Most imortantly, citing the original author is mandatory.