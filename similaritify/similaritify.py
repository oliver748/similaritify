# - # - # - #

import sys
sys.path.append('similaritify')

# - # - # - #

from methods.AKAZE import AKAZE
from methods.SSIM import SSIM
from methods.ORB import ORB

import cv2
import math


class Similaritify:
    def __init__(self, target_width):
        self.target_width = target_width

    def load_images(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None:
            raise FileNotFoundError(f"Cannot open/read file \"{img1_path}\". Please check if file path is correct.")
        if img2 is None:
            raise FileNotFoundError(f"Cannot open/read file \"{img2_path}\". Please check if file path is correct.")

        return img1, img2

    def resize_images(self, img1, img2):
        def get_simplified_aspect_ratio(img):
            height, width = img.shape[:2]
            gcd = math.gcd(width, height)
            return width // gcd, height // gcd
                
        img1_ar = get_simplified_aspect_ratio(img1)
        img2_ar = get_simplified_aspect_ratio(img2)

        assert img1_ar == img2_ar, "The images has to have the same aspect ratio."

        target_height = int(self.target_width * img1_ar[0] / img1_ar[1])

        img1 = cv2.resize(img1, (self.target_width, target_height))
        img2 = cv2.resize(img2, (self.target_width, target_height))

        return img1, img2

    def run(self, image_1, image_2, methods, need_resizing):
        
        img1, img2 = self.load_images(image_1, image_2)

        if need_resizing:
            img1, img2 = self.resize_images(img1, img2)

        results_dic = {}

        for method in methods:
            if method.lower() == "akaze":
                akaze = AKAZE()
                result = akaze.compare(img1, img2)
                results_dic["AKAZE"] = result
            
            if method.lower() == "ssim":
                ssim = SSIM()
                result = ssim.compare(img1, img2)
                results_dic["SSIM"] = result
            
            if method.lower() == "orb":
                orb = ORB()
                result = orb.compare(img1, img2)
                results_dic["ORB"] = result
                
        

        return results_dic