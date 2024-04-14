from skimage.metrics import structural_similarity
import cv2
import numpy as np



class SSIM:
    def __init__(self, target_width, resize_images):
        self.target_width = target_width
        self.resize_images = resize_images

    def load_images(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None:
            raise FileNotFoundError(f"Cannot open/read file \"{img1_path}\". Please check if file path is correct.")
        if img2 is None:
            raise FileNotFoundError(f"Cannot open/read file \"{img2_path}\". Please check if file path is correct.")

        #self.print_debug("Loaded images", self.start_time)

        return img1, img2

    def resize(self, img1, img2):
        # Calculate target height to maintain 16:9 aspect ratio
        target_height = int(self.target_width * 9 / 16)

        # Resize images to 1280x720 while maintaining aspect ratio
        img1 = cv2.resize(img1, (self.target_width, target_height))
        img2 = cv2.resize(img2, (self.target_width, target_height))

       # self.print_debug(f"Resized img1 to {img1.shape[1]}x{img1.shape[0]} and img2 to {img2.shape[1]}x{img2.shape[0]}", self.start_time)

        return img1, img2


    def compare(self, img1_path, img2_path):

        img1, img2 = self.load_images(img1_path, img2_path)

        if self.resize_images:
            img1, img2 = self.resize(img1, img2)

        # Convert images to grayscale
        img1_gs = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gs = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Compute SSIM between two images
        (score, diff) = structural_similarity(img1_gs, img2_gs, full=True)
        
        return score*100