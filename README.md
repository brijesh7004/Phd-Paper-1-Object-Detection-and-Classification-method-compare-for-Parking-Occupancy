# Parking Space Detection Using Computer Vision and Deep Learning

This project aims to enhance parking space detection through computer vision, deep learning, and a novel boundary comparison method. Multiple state-of-the-art algorithms are evaluated for accuracy and efficiency, including detection models (RetinaNet, SSD, YOLO8, YOLO9) and classification models (ResNet50, VGG16, AlexNet, MobileNet).

## Features

- **Boundary Configuration**  
  Use `Config_Generator.py` to load a video, extract the first frame, and manually draw parking boundaries using four points. Multiple boundaries can be added, and the configuration can be saved for later use.

- **Detection & Classification Testing**  
  Run `Parking_Occupation_Detector_Using_Classification.py` and `Parking_Occupation_Detector_Using_Detection.py` to test different models.  
  - Load the saved configuration and video.  
  - Select the desired model via Python GUI.  
  - View real-time detection results and save processed images.  
  - Frames are skipped during processing to optimize speed for testing.

## Usage

### Step 1: Configuring Parking Boundaries
1. Run `Config_Generator.py`.  
2. Load a video and draw parking boundaries on the first frame using four points.  
3. Save the configuration file for future detection tasks.

### Step 2: Testing Detection & Classification
1. Run either `Parking_Occupation_Detector_Using_Classification.py` or `Parking_Occupation_Detector_Using_Detection.py`.  
2. Load the saved configuration and select a model through the GUI.  
3. Observe real-time detection results and processed images.  
4. Frames are processed with frame skipping for faster testing.

## Note
- The system is designed for flexibility and speed, allowing rapid testing of multiple models on custom parking lot configurations.

---

For any questions or contributions, please contact the repository maintainer.
