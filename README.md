# Face Detection

Implements Google machine learning algorithm to detect faces on your screen or camera.

Using [mediapipe](https://google.github.io/mediapipe/) state-of-the-art face detection built with deep learning.
Mediapipe allows for end-to-end acceleration, it has ready-to-use solutions and it is free and open source! Best of all worlds!


* Easy to implement into your code
* Organized and clean code to allow for enhanced readability 
* Separate Thread to prevent blocking

## How to use

Webcam
```python
import face_detection
face_detection.Video(confidence_input=0.5, draw=True, camera=True)
```
Monitor
```python
import face_detection
face_detection.Video(confidence_input=0.5, draw=True, camera=False, dimensions=(50, 60, 500,500))
```

## Images
<img src="images/faces1.png" width="200" >
