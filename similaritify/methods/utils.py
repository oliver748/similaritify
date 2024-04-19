import cv2

def convert_to_grayscale(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return gray1, gray2

def find_keypoints_and_descriptors(detector, image):
    """ Find keypoints and descriptors with ORB. """
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors

def create_matcher(norm_type=None, cross_check=False):
    return cv2.BFMatcher(normType=norm_type, crossCheck=cross_check)

def find_valid_matches(matches, distance_modifier):
    if matches is None:
        return None
    
    valid_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < distance_modifier * n.distance:
                valid_matches.append([m])
                
    return valid_matches

def default_match_descriptors(matcher, descriptors1, descriptors2, need_sort=True):
    """ Match descriptors using a brute force matcher. """
    matches = matcher.match(descriptors1, descriptors2)
    if need_sort: 
        matches = sorted(matches, key=lambda x: x.distance)
    return matches

def knn_match_descriptors(matcher, descs1, descs2, k_val=2):
    return matcher.knnMatch(descs1, descs2, k=k_val)


def calculate_match_percentage(matches, keypoints1):
    """ Calculate the match percentage. """
    if not keypoints1:
        return 0
    match_percentage = (len(matches) / len(keypoints1))
    return match_percentage