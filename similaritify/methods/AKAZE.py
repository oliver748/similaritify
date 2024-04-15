import cv2
import time
import os

class AKAZE:
    def __init__(self, distance_modifier=0.75):
        # dir should be relative to ../image_pairer
        self.detector = cv2.AKAZE_create()
        self.working_dir = os.path.join(os.path.dirname(__file__))
        self.distance_modifier = distance_modifier
        
    def convert_to_grayscale(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return gray1, gray2

    def detect_and_compute(self, gray1, gray2):        
        kps1, descs1 = self.detector.detectAndCompute(gray1, None)
        kps2, descs2 = self.detector.detectAndCompute(gray2, None)

        if kps1 is None or descs1 is None:
            return None, None, None, None
        elif kps2 is None or descs2 is None:
            return None, None, None, None
        
        return kps1, descs1, kps2, descs2

    def match_descriptors(self, descs1, descs2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(descs1, descs2, k=2)
        return matches

    def find_valid_matches(self, matches):
        if matches is None:
            return None
        
        valid_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < self.distance_modifier * n.distance:
                    valid_matches.append([m])
                    
        return valid_matches

    def calculate_match_percentage(self, kps1, kps2, valid_matches):
        if valid_matches is None:
            return 0
        if len(kps1) == 0:
            return 0
        else:
            match_percentage = (len(valid_matches) / len(kps1))
            return match_percentage

    def compare(self, img1, img2):
        # Convert rgb images to grayscale
        gray1, gray2 = self.convert_to_grayscale(img1, img2)

        # Detect and compute keypoints and descriptors
        kps1, descs1, kps2, descs2 = self.detect_and_compute(gray1, gray2)

        # If no keypoints or descriptors are found, return 0 (match percentage is 0)
        if any(v is None for v in [descs1, descs2, kps1, kps2]):
            return None
        
        # Match descriptors
        matches = self.match_descriptors(descs1, descs2)

        # Filter valid matches
        valid_matches = self.find_valid_matches(matches)

        return self.calculate_match_percentage(kps1, kps2, valid_matches)
