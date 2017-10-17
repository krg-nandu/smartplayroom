import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

def method1(im1,im2):
	# Convert images to grayscale
	im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
	im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
 
	# Find size of image1
	sz = im1.shape
 
	# Define the motion model
	warp_mode = cv2.MOTION_TRANSLATION
 
	# Define 2x3 or 3x3 matrices and initialize the matrix to identity
	if warp_mode == cv2.MOTION_HOMOGRAPHY :
	    warp_matrix = np.eye(3, 3, dtype=np.float32)
	else :
	    warp_matrix = np.eye(2, 3, dtype=np.float32)
 
	# Specify the number of iterations.
	number_of_iterations = 5000;
	 
	# Specify the threshold of the increment
	# in the correlation coefficient between two iterations
	termination_eps = 1e-10;
	 
	# Define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
	# Run the ECC algorithm. The results are stored in warp_matrix.
	(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
 
	if warp_mode == cv2.MOTION_HOMOGRAPHY :
	    # Use warpPerspective for Homography 
	    im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	else :
	    # Use warpAffine for Translation, Euclidean and Affine
	    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

	# Show final results
	cv2.imshow("Image 1", im1)
	cv2.imshow("Image 2", im2)
	cv2.imshow("Aligned Image 2", im2_aligned)
	cv2.waitKey(0)

def deskew(orig_image,skewed_image,M):
    im_out = cv2.warpPerspective(skewed_image, np.linalg.inv(M), (orig_image.shape[1], orig_image.shape[0]))
    
    fig = plt.figure()
    ax = []
    ax.append( fig.add_subplot(311) )
    ax.append( fig.add_subplot(312) )
    ax.append( fig.add_subplot(313) )

    ax[0].imshow(orig_image)
    ax[0].set_title('Template')

    ax[1].imshow(skewed_image)
    ax[1].set_title('Frame to be mapped!')

    ax[2].imshow(im_out)
    ax[2].set_title('Estimated Affine Transformation')

    plt.show()

def method2(orig_image,skewed_image):
	surf = cv2.xfeatures2d.SURF_create(400)
	kp1, des1 = surf.detectAndCompute(orig_image, None)
	kp2, des2 = surf.detectAndCompute(skewed_image, None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m, n in matches:
    		if m.distance < 0.7 * n.distance:
	        	good.append(m)

	MIN_MATCH_COUNT = 10
	if len(good) > MIN_MATCH_COUNT:
	    src_pts = np.float32([kp1[m.queryIdx].pt for m in good
        	                  ]).reshape(-1, 1, 2)
	    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
        	                  ]).reshape(-1, 1, 2)

	    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

	    # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
	    ss = M[0, 1]
	    sc = M[0, 0]
	    scaleRecovered = math.sqrt(ss * ss + sc * sc)
	    thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
	    print("Calculated scale difference: %.2f\nCalculated rotation difference: %.2f" % (scaleRecovered, thetaRecovered))	
	    deskew(orig_image,skewed_image,M)	
	else:
	    print("Not  enough  matches are found   -   %d/%d" % (len(good), MIN_MATCH_COUNT))
	    matchesMask = None

# Read the images to be aligned
im1 =  cv2.imread("image1.png");
im2 =  cv2.imread("image2.png");

method2(im1,im2)

