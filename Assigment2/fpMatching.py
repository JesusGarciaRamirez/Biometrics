import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
from Evaluation import Metrics

def is_defined(instance,argument):
    if(getattr(instance,argument)==None):
        raise ValueError

class fpMatcher(object):
    def __init__(self,dataset_path,keypoint_extractor,detector_opts,distance_metric):
        self.dataset_path = dataset_path
        self.keypoint_extractor = keypoint_extractor
        self.matcher = cv2.BFMatcher(distance_metric, crossCheck=True)
        self.detector = self.keypoint_extractor(**detector_opts)
        self.images = None
        self.labels = None
        self.masks = None

    def sample_pairs(self,genuine=True):
        """Sample pair of random test images from dataset:
            if genuine = True, sample two images from same individual
            otherwise, it present images from different individuals"""
        try:
            is_defined(self,"images")
        except:
            self.get_enhanced_images()
        #Random sample
        idx1=np.random.randint(0,len(self.labels)-1)
        #get label of random instance
        label_1=self.labels[idx1]
        if genuine:
            #get random instance with label==label_1
            idx_list = np.squeeze(np.argwhere(self.labels == label_1)) #genuine indexes
        else:
            #get random instance with label!=label_1
            idx_list = np.squeeze(np.argwhere(self.labels != label_1)) #impostor indexes
        #sample one index from correspondant list(genuine or impostors)
        idx2=int(np.random.choice(idx_list, 1))
        return [idx1,idx2]

    def get_samples_info(self,idx_list):
        images=[self.images[idx] for idx in idx_list]
        masks=[self.masks[idx] for idx in idx_list]
        labels=[self.labels[idx] for idx in idx_list]
        return images,masks,labels
        
    def get_enhanced_images(self):
        p_file = open(self.dataset_path, "rb" )
        [images, labels, masks] = pickle.load(p_file)
        self.images=images
        self.labels=np.array(labels,dtype=np.int32)
        self.masks=masks

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

    def keypoint_plot(self,images,masks,labels=None):
        """Plot keypoint extraction results on test image"""
        ax = plt.figure()
        _, ax = plt.subplots(nrows = 1, ncols = len(images), figsize=(10,10))
        for i,(img,mask) in enumerate(zip(images,masks)):
            #compute keypoints
            kp,_ = self.detect_kp(img,mask)
            #draw keypoints on image
            kp_result_image = cv2.drawKeypoints(img,
                                                 kp, None, (255, 0, 0), 
                                                 cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            # Add to correspondant matplotlib subplot
            ax[i].imshow(kp_result_image)
            if(labels is not None):
                ax[i].set_title(f"label of image : {labels[i]}")
            
        
        plt.show()

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
    
    def local_scoring(self,images,masks,score_fcn,**kwargs):
        #Local matching
        matches, _ = self.match_BruteForce_local(images,masks)
        #score
        score = score_fcn(matches,**kwargs)
        return score

class fpScorer(fpMatcher):

    def __init__(self,dataset_path,keypoint_extractor,detector_opts,distance_metric):
        super(fpScorer,self).__init__(dataset_path,keypoint_extractor,detector_opts,distance_metric)

    def local_score_dataset(self,downsampling=True,**kwargs):
        try:
            is_defined(self,"images")
        except:
            self.get_enhanced_images()

        
        if(downsampling):
            #get reduced set of total dataset (eg. 30 samples)
            index_list = list(np.random.choice(len(self.images), 30,replace=False))
        else:
            index_list = range(len(self.images))
        genuine_ids=[]
        scores=[]   
        for count,idx1 in tqdm(enumerate(index_list)):
            rest_idx = index_list[count+1:]
            for idx2 in rest_idx:
                #get samples info
                images,masks,labels= self.get_samples_info([idx1,idx2])
                #local matching
                matches,_ = self.match_BruteForce_local(images,masks)
                #score
                score = self.local_scoring(matches,**kwargs)
                #update
                scores.append(score)
                label = 1 if labels[0]==labels[1] else 0
                genuine_ids.append(label) 

        scores = np.array(scores,dtype=np.float32)
        genuine_ids = np.array(genuine_ids,dtype=np.int32)
        return scores,genuine_ids

    def local_scoring(self,matches,N_best):
        #get kp distances from top N best matches
        best_matches_distances=np.array([matches[idx].distance 
                                        for idx in range(N_best)],dtype=np.float32)
        #local score == Inverse of mean distances 
        score = 1/ (np.mean(best_matches_distances) + 1)
        return score

#Debugging
if __name__ == "__main__":
    #read images,masks..
    dataset_path= "./fprdata/DB1_enhanced.p"
    p_file = open(dataset_path, "rb" )
    [images_enhanced_db1, labels_db1, masks_db1] = pickle.load(p_file)
    #test images from same individual
    #create fpMatcher instance
    # detector_opts={"nfeatures": 500}
    # detector_opts={"hessianThreshold ": 400}
    detector_opts={}
    # detector=cv2.xfeatures2d.SURF_create
    # detector=cv2.KAZE_create
    # detector=cv2.AKAZE_create
    detector=cv2.BRISK_create
    # detector=cv2.ORB_create
    # fpmatcher=fpMatcher(dataset_path,detector,detector_opts,cv2.NORM_HAMMING)
    #local scoring
    scoring_otps={"N_best": 10}
    # scores,ids = fpmatcher.create_dataset_scores(test_local_scoring,**scoring_otps)
    fpscorer=fpScorer(dataset_path,detector,detector_opts,cv2.NORM_HAMMING)
    scores,labels = fpscorer.local_score_dataset(downsampling=False,**scoring_otps)
    metric_logger = Metrics(scores,labels,"FingerPrint_local")
    _,ax=plt.subplots(1,1)
    metric_logger.plot_roc_curve(ax)
    plt.show()
    print("Hi")
    pass