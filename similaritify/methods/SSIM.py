import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim



class SSIM:
    def __init__(self, multi_channel=True, win_size=None):
        self.multi_channel = multi_channel
        self.win_size = win_size

    def compare(self, img1, img2):
        if not self.multi_channel:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # When multichannel is True, ensure the channel axis is correctly specified (depends on grayscale image or rgb)
        channel_axis = -1 if self.multi_channel else None

        mssim, _ = ssim(img1, img2, multichannel=self.multi_channel, full=True, win_size=self.win_size, channel_axis=channel_axis)
        return mssim