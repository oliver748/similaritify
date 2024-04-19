import sys
import os

sys.path.append(os.path.join("similaritify", "methods"))

import cv2
import numpy as np
import utils as ut


class ORB:
    def compare(self, img1, img2, num_features=500):
        orb = cv2.ORB_create(nfeatures=num_features)
        matcher = ut.create_matcher(norm_type=cv2.NORM_HAMMING, cross_check=True)
        kps1, descs1 = ut.find_keypoints_and_descriptors(detector=orb, image=img1)
        kps2, descs2 = ut.find_keypoints_and_descriptors(detector=orb, image=img2)
        
        if any(v is None for v in [descs1, descs2, kps1, kps2]):
            return None
        
        matches = ut.default_match_descriptors(matcher, descs1, descs2, need_sort=True)

        return ut.calculate_match_percentage(matches, kps1)
