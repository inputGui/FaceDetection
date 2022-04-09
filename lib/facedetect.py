import numpy as np
import mss
import mss.tools
import cv2
import time
import mediapipe as mp
from threading import Thread

"""
Personal assignment: Find faces in video.
"""


class FaceDetector:
    def __init__(self, minDetectionConf=0.5, draw=True):
        self.minDetectionCon = minDetectionConf
        self.mpfacedetection = mp.solutions.face_detection
        self.mpdraw = mp.solutions.drawing_utils
        #  Initialize FaceDetection()
        self.facedetection = self.mpfacedetection.FaceDetection(self.minDetectionCon)
        self.draw = True

    def find_faces(self, img):
        img = np.flip(img[:, :, :3], 2)  # 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facedetection.process(img)
        bboxes = []
        #  Extract information from results
        # Check if results available
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                image_h, image_w, image_c = img.shape
                bbox = int(bboxC.xmin * image_w), int(bboxC.ymin * image_h), \
                       int(bboxC.width * image_w), int(bboxC.height * image_h)
                bboxes.append([bbox, detection.score])
                if self.draw:
                    img = self.fancy_draw(img, bbox)
                    cv2.putText(img, f"{int(detection.score[0] * 100)}%", (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

        return img, bboxes

    def fancy_draw(self, img, bbox, lenght=30, thickness=5, rt=1):
        # Origin
        x, y, w, h = bbox
        # Bottom right
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top left x,y
        cv2.line(img, (x, y), (x + lenght, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y + lenght), (255, 0, 255), thickness)

        # Top right
        cv2.line(img, (x1, y), (x1 - lenght, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y + lenght), (255, 0, 255), thickness)

        # Bottom left
        cv2.line(img, (x, y1), (x + lenght, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1 - lenght), (255, 0, 255), thickness)

        # Bottom right
        cv2.line(img, (x1, y1), (x1 - lenght, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - lenght), (255, 0, 255), thickness)

        return img
    # Tell the other worker to stop


class Video(Thread):
    def __init__(self, confidence_input, draw=True, camera=False, dimensions=(0, 40, 1000, 1300)):
        Thread.__init__(self)
        self.start()
        self.face_detector = FaceDetector(confidence_input, draw=draw)
        self.dimensions = dimensions
        self.camera = camera
        self.draw = draw
        self.monitor_cap = None
        self.face_cap = None

        if self.camera:
            self.capture_cam(draw=self.draw)
        else:
            self.capture_monitor(draw=self.draw)

    def capture_cam(self, draw: bool = True):
        self.face_cap = cv2.VideoCapture(0)
        pTime = 0
        while True:
            ret, frame = self.face_cap.read()
            if ret:
                img, bboxes = self.face_detector.find_faces(frame)
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
                cv2.imshow('Video', img)

            if cv2.waitKey(15) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

        self.face_cap.release()
        cv2.destroyAllWindows()

    def capture_monitor(self, draw: bool = True):
        self.monitor_cap = mss.mss()
        pTime = 0
        while True:
            img = np.asarray(self.monitor_cap.grab(self.dimensions))
            img, bboxes = self.face_detector.find_faces(img)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
            cv2.imshow('Video', img)

            if cv2.waitKey(15) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

        self.monitor_cap.close()
        cv2.destroyAllWindows()


#image = Video(0.5, True, True)
