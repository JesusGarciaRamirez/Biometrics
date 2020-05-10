import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm

class fpMatcher(object):
    def __init__(self,dataset_path,keypoint_extractor,detector_opts,distance_metric):
        self.dataset_path = dataset_path
        self.keypoint_extractor = keypoint_extractor
        self.matcher = cv2.BFMatcher(distance_metric, crossCheck=True)
        self.detector = self.keypoint_extractor(**detector_opts)

    def is_defined(self,argument):
        if(getattr(self,argument)==None):
            raise ValueError

    def get_enhanced_images(self):
        p_file = open(self.dataset_path, "rb" )
        [images_enhanced, labels, masks] = pickle.load(p_file)
        return images_enhanced,labels,masks

    def detect_raw_kp(self,image):
        kp, desc = self.detector.detectAndCompute(image, None)
        return kp, desc

    def _remove_edge_kps(self,mask, kp, desc):
        mask_b = mask.astype(np.uint8)  #convert to an unsigned byte
        # morphological erosion
        mask_b*=255
        mask_e = cv2.erode(mask_b, kernel = np.ones((5,5),np.uint8), iterations = 5)
        # remove keypoints and their descriptors that lie outside this eroded mask
        kpn = [kp[i] for i in range(len(kp)) if mask_e.item(int(kp[i].pt[1]),int(kp[i].pt[0])) == 255]
        descn = np.vstack([desc[i] for i in range(len(kp)) if mask_e.item(int(kp[i].pt[1]),int(kp[i].pt[0])) == 255])
        return kpn, descn

    def detect_kp(self,image,mask):
        kp,desc = self.detect_raw_kp(image) #raw kp and desc
        kp,desc = self._remove_edge_kps(mask,kp,desc) #remove low quality kps
        return kp, desc

    def match_BruteForce_local(self, images, masks):    
        """Brute Force all pair matcher: returns all pairs of best matches
            depending on type of descriptor use the corresponding norm
            crossCheck=True only retains pairs of 
            keypoints that are each other best matching pair"""
        #get descriptors
        kp1,des1 = self.detect_kp(images[0],masks[0])
        kp2,des2 = self.detect_kp(images[1],masks[1])
        keypoints = [kp1,kp2]
        #get matches
        matches = self.matcher.match(des1, des2)
        # sort matches based on feature distance
        matches.sort(key=lambda x: x.distance, reverse=False)
        return matches,keypoints
    
    def transform_keypoints(self,key_points, transformation_matrix):
        # convert keypoint list to Nx1x2 matrix
        mat_points = cv2.KeyPoint.convert(key_points).reshape(-1,1,2)
        # transform points 
        mat_reg_points = cv2.transform(mat_points, transformation_matrix)
        # return transformed keypoint list
        return cv2.KeyPoint.convert(mat_reg_points)   

    def match_BruteForce_global(self,images,masks,good_match_percent = 0.75):
        #Local matching
        matches,keypoints = self.match_BruteForce_local(images,masks)
        # select the best x percent best matches (on local feature vector level) for further global comparison
        GOOD_MATCH_PERCENT = good_match_percent
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        good_matches = matches[:numGoodMatches]
    
        # retain only the keypoints associated to the best matches 
        src_pts = np.float32([keypoints[0][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[1][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # estimate an optimal 2D affine transformation with 4 degrees of freedom,
        # limited to combinations of translation, rotation, and uniform scaling
        
        # this is the core of the global consistency check: if we find the correct transformation
        # (which we expect for genuine pairs and not for imposter pairs), we can use it as an
        # additional check by verifying the geometrical quality of the match
        
        # M stores the optimal transformation
        # inliers stores the indices of the subset of points that were finally used to calculate the optimal transformation
        
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method =  cv2.RANSAC, 
                                                confidence = 0.9, ransacReprojThreshold = 10.0, 
                                                maxIters = 5000, refineIters = 10)

        # get the inlier matches
        matched = [x for x,y in zip(good_matches, inliers) if y[0] == 1]

        # The optimal transformation is only correct for genuine pairs in about 75% of cases (experimentally on dataset DB1).
        # One can build additional checks about the validity of the transformation,
        # e.g. too large translations, rotations and/or scale factors
        
        # A simple one is to test the number of keypoints that were used in calculating the transformation. 
        # If this number is is too small, then the transformation is most possibly unreliable. 
        # In that case, we reset the transformation to the identity
        if np.sum(inliers) < 5:
            M = np.eye(2, 3, dtype=np.float32)

        # transform the first keypoint set using the transformation M
        kp1_reg = self.transform_keypoints(keypoints[0], M)
                                                
        return kp1_reg, matched, M, keypoints

    def local_matching_plot(self,images, masks):
        #Local matching
        matches,keypoints = self.match_BruteForce_local(images,masks)
        # show the result using drawMatches
        imMatches = cv2.drawMatches(images[0],keypoints[0],
                                    images[1],keypoints[1],matches, None) 
        plt.imshow(imMatches)
        plt.show()
    
    def global_matching_plot(self, images, masks):
        #global matching
        kp1_reg, matched, M, keypoints = self.match_BruteForce_global(images,masks)
        # visualization of the matches after affine transformation
        height, width = images[1].shape[:2]
        im1Reg = cv2.warpAffine(images[0], M, (width, height))

        # only the inlier matches (matched) are shown
        imMatches = cv2.drawMatches(im1Reg, kp1_reg,
                                    images[1], keypoints[1], matched, None) #, flags=2)
        plt.imshow(imMatches)
        plt.show()
        print("Number of affine inliers :{}".format(len(matched)))

