import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_confidence=0.5, model_selection=1):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence
        )

    def detect_faces(self, image):
        """
        Detects faces and returns safe integer bounding boxes [(x, y, w, h), ...].
        """
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)
        boxes = []

        if results and results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = min(int(bbox.width * w), w - x)
                height = min(int(bbox.height * h), h - y)

                if width > 0 and height > 0:
                    boxes.append((x, y, width, height))

        return boxes

    def close(self):
        if hasattr(self, "detector"):
            self.detector.close()