import cv2
import numpy as np
class ORB:
    def initialize_orb(self, num_features=500):
        """ Initialize ORB detector. """
        orb = cv2.ORB_create(nfeatures=num_features)
        return orb

    def find_keypoints_and_descriptors(self, orb, image):
        """ Find keypoints and descriptors with ORB. """
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_descriptors(self, descriptors1, descriptors2):
        """ Match descriptors using a brute force matcher. """
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def calculate_match_percentage(self, matches, keypoints1):
        """ Calculate the match percentage. """
        if not keypoints1:
            return 0
        match_percentage = (len(matches) / len(keypoints1))
        return match_percentage

    def compare(self, img1, img2):
        orb = self.initialize_orb()
        keypoints1, descriptors1 = self.find_keypoints_and_descriptors(orb, img1)
        keypoints2, descriptors2 = self.find_keypoints_and_descriptors(orb, img2)
        matches = self.match_descriptors(descriptors1, descriptors2)
        return self.calculate_match_percentage(matches, keypoints1)