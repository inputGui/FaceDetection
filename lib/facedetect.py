import numpy as np
import mss
import mss.tools
import cv2
import time
import mediapipe as mp

"""
Find faces in video.
"""

confidence_input = None  # Placeholder for minimum confidence


class FaceDetector:
    def __init__(self, minDetectionConf=0.5):
        self.minDetectionCon = minDetectionConf
        self.mpfacedetection = mp.solutions.face_detection
        self.mpdraw = mp.solutions.drawing_utils
        #  Initialize FaceDetection()
        self.facedetection = self.mpfacedetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
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
                if draw:
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


def main():
    """
    Records and scans rect of screen
    Displays to the user any detection of face
    """
    rect = (0, 40, 1000, 1300)
    sct = mss.mss()
    pTime = 0
    detector = FaceDetector(confidence_input)
    while True:
        img = np.asarray(sct.grab(rect))
        img, bboxes = detector.findFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
        cv2.imshow('Video', img)

        if cv2.waitKey(15) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

    sct.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    confidence_input = float(input("Input desired minimum confidence level ( 0.1 - 0.9 ):"))
    main()

