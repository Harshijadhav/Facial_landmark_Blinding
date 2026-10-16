import cv2
import mediapipe as mp


class LandmarkDetector:
    """
    MediaPipe Face Mesh detector wrapper configured for high-precision
    facial landmark extraction and alignment evaluation.
    """

    def __init__(self, min_confidence=0.65, max_faces=1):
        self.min_confidence = min_confidence
        self.max_faces = max_faces
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=self.max_faces,
            refine_landmarks=True,
            min_detection_confidence=self.min_confidence
        )

    def detect_landmarks(self, image_rgb):
        """Processes an RGB image and returns the raw MediaPipe result object."""
        return self.face_mesh.process(image_rgb)

    def extract_points(self, image_shape, results):
        """Helper method to extract integer pixel coordinates for primary face."""
        if not results or not results.multi_face_landmarks:
            return []

        h, w = image_shape[:2]
        primary_face = results.multi_face_landmarks[0]
        points = [(int(pt.x * w), int(pt.y * h)) for pt in primary_face.landmark]
        return points

    def close(self):
        if hasattr(self, "face_mesh"):
            self.face_mesh.close()