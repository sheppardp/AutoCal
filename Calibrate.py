import numpy as np
import cv2
from numpy import linalg as npl

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
		
		if M is not None:
			matchesMask = mask.ravel().tolist()
			print(M)
			h,w = grayImage1.shape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			dst = cv2.perspectiveTransform(pts,M)

			#matchImage = cv2.polylines(image1.copy(),[np.int32(dst)],True,255,3, cv2.LINE_AA)
			
			draw_params = dict(matchColor = (0,255,0), # draw matches in green color
						singlePointColor = None,
						matchesMask = matchesMask, # draw only inliers
						flags = 2)

			matchImage = cv2.drawMatches(image1.copy(),keypoint1,image2.copy(),keypoint2,good,None,**draw_params)
			
			#print("Homography: ",M)

			criticalPoints = []
			eigenvalues, eigenvectors = npl.eig(M)

			np.set_printoptions(precision=5, suppress=True, linewidth=200)

			#print ("Eigenvalues: ",eigenvalues)
			print ("Eigenvectors: ",eigenvectors)

			for i in range(3):
				eigenvector = eigenvectors[:,i]
				if (abs(eigenvector.imag) > .01).any():				
					#criticalPoints.append(eigenvector[1:3]/eigenvector[0])
					criticalPoints.append(eigenvector[0:2]/eigenvector[2])
							
			#print("Circular points: ",criticalPoints)
			
			warp = cv2.warpPerspective(image1.copy(), M, (image2.shape[1], image2.shape[0]))
			
			print(matchImage.shape)
			print(warp.shape)
			
			output = np.zeros((matchImage.shape[0]+warp.shape[0], matchImage.shape[1], 3))
			
			print(output.shape)
			
			output[0:matchImage.shape[0], 0:matchImage.shape[1]] = matchImage
			output[matchImage.shape[0]:matchImage.shape[0]+warp.shape[0], 0:warp.shape[1]] = image2-warp
			output[matchImage.shape[0]:matchImage.shape[0]+warp.shape[0], warp.shape[1]:2*warp.shape[1]] = warp

			return [criticalPoints, output, M]
			
		else:
			print("No valid homography found")

	else:
		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None
		return []	

def getConic(points):
	M = []
	
	for point in points:
		M.append(np.array([point[0]**2, point[0]*point[1], point[1]**2, point[0], point[1], 1]))
	
	return np.vstack(M)	
	
def getFocalLengthConic(points):
	M = []
	
	for point in points:
		M.append(np.array([point[0]**2, point[1]**2, 1]))
	
	return np.vstack(M)	
	
def getFocalLength(A):
	
	u,s,v = npl.svd(A)

	
	conic = v[-1,:]	
	
	a=conic[0]
	c=conic[1]
	f=conic[2]

	iac = np.array([[a, 0, 0],[0, c, 0],[0, 0, f]])
	iac /= f
	
	print("iac: ",iac)

	diac = npl.inv(iac)

	#print(diac)

	chol = npl.cholesky(diac)

	K = chol.T.conj()
	
	K /= K[2][2]

	return K
		
def getCameraParameters(A):
	
		u,s,v = npl.svd(A)

		print("u: ",u)
		print("s: ",s)
		print("v: ",v)

		# Get right singular vector (rows in numpy svd)
		conic = v[-1,:]
		
		#print("Conic: ", conic)
		#print("Mult: " , np.dot(A,conic))

		a=conic[0]
		b=conic[1]
		c=conic[2]
		d=conic[3]
		e=conic[4]
		f=conic[5]

		iac = np.array([[a, b/2, d/2],[b/2, c, e/2],[d/2, e/2, f]])
		
		iac /= f
		
		#print("iac: ",iac)

		diac = npl.inv(iac)

		#print(diac)

		chol = npl.cholesky(diac)

		K = chol.T.conj()
		
		K /= K[2][2]

		return K








