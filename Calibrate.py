import numpy as np
import cv2
from numpy import linalg as LA

MIN_MATCH_COUNT = 6

def getCriticalPoints(image1, image2):
	# Convert input images to grayscale
	grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	grayImage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
	
	# build the sift keypoint detector
	sift = cv2.xfeatures2d.SIFT_create()
	
	# Detect Keypoints and calculate descriptors
	keypoint1, descriptor1 = sift.detectAndCompute(grayImage1,None)
	keypoint2, descriptor2 = sift.detectAndCompute(grayImage2,None)

	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary

	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(descriptor1,descriptor2,k=2)
	
	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
			
	if len(good)>=MIN_MATCH_COUNT:
		src_pts = np.float32([ keypoint1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ keypoint2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()
		
		h,w = grayImage1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)

		#matchImage = cv2.polylines(image1.copy(),[np.int32(dst)],True,255,3, cv2.LINE_AA)
		
		draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					singlePointColor = None,
					matchesMask = matchesMask, # draw only inliers
					flags = 2)

		matchImage = cv2.drawMatches(image1.copy(),keypoint1,image2.copy(),keypoint2,good,None,**draw_params)
		
		print(M)

		criticalPoints = []
		eigenvalues, eigenvectors = LA.eig(M)

		np.set_printoptions(precision=5, suppress=True, linewidth=200)

		print ("Eigenvalues: ",eigenvalues)
		print ("Eigenvectors: ",eigenvectors)

		for i in range(3):
			eigenvector = eigenvectors[:,i]
			if (abs(eigenvector.imag) > .01).any():				
				criticalPoints.append(eigenvector[1:3]/eigenvector[0])
						
		print("Circular points: ",criticalPoints)
		
		return criticalPoints, matchImage

	else:
		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None
		return []	

def getConic(points):
	M = []
	
	for point in points:
		M.append(np.array([point[0]**2, point[0]*point[1], point[1]**2, point[0], point[1], 1]))
	
	return np.vstack(M)	
		

#critical1 = getCriticalPoints(cv2.imread("orbit1_0.jpg"), cv2.imread("orbit1_3.jpg"))
#critical2 = getCriticalPoints(cv2.imread("orbit2_0.jpg"), cv2.imread("orbit2_3.jpg"))
#critical3 = getCriticalPoints(cv2.imread("orbit3_0.jpg"), cv2.imread("orbit3_3.jpg"))


#A = getConic(critical1+critical2+[critical3[0]])


#u,s,v = LA.svd(A)

#conic = v[-1,:]

#a=conic[0]
#b=conic[1]
#c=conic[2]
#d=conic[3]
#e=conic[4]
#f=conic[5]

#iac = np.array([[a, b/2, d/2],[b/2, c, e/2],[d/2, e/2, f]])

#diac = LA.inv(iac)

#print(diac)

#K = LA.cholesky(diac)

#print(K)









