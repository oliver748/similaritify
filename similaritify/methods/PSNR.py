import sys
import os

sys.path.append(os.path.join("similaritify", "methods"))

from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
import utils as ut

class PSNR:
    def compare(self, img1, img2):
        gray1, gray2 = ut.convert_to_grayscale(img1, img2)

        # if mse is 0, then images are identical
        if mean_squared_error(gray1, gray2) == 0:
            return 1
        
        return peak_signal_noise_ratio(gray2, gray1)