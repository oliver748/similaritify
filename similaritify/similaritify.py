# - # - # - #

import sys
sys.path.append('similaritify')

# - # - # - #

from methods.AKAZE import AKAZE
from methods.SSIM import SSIM


def run(image_1, image_2, methods, target_width, resize_images):

    results_dic = {}

    for method in methods:
        if method.lower() == "akaze":
            akaze = AKAZE(target_width=target_width, resize_images=resize_images)
            result = akaze.compare(image_1, image_2)
            results_dic["AKAZE"] = result
        
        if method.lower() == "ssim":
            ssim = SSIM(target_width=target_width, resize_images=resize_images)
            result = ssim.compare(image_1, image_2)
            results_dic["SSIM"] = result
    

    return results_dic