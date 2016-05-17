import numpy as np
import cv2
from numpy import linalg as npl

# Minimum number of matched features for homography calculation
MIN_MATCH_COUNT = 6

# Given two images, feature match and find the homography
def getCircularPoints(image1, image2):
	
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

	# Run the matcher
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(descriptor1,descriptor2,k=2)
	
	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
			
	if len(good)>=MIN_MATCH_COUNT:
		
		# Find the homography
		src_pts = np.float32([ keypoint1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ keypoint2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		
		if M is not None:
			matchesMask = mask.ravel().tolist()
			draw_params = dict(matchColor = (0,255,0), # draw matches in green color
						singlePointColor = None,
						matchesMask = matchesMask, # draw only inliers
						flags = 2)

			# Show the matches
			matchImage = cv2.drawMatches(image1.copy(),keypoint1,image2.copy(),keypoint2,good,None,**draw_params)
			
			# Find the eigenvectors of the homography
			eigenvalues, eigenvectors = npl.eig(M)
			np.set_printoptions(precision=5, suppress=True, linewidth=200)
			print ("Eigenvectors: ",eigenvectors)

			# Use the eigenvectors to resolve the circular points
			circularPoints = []
			for i in range(3):
				# Each row in the eigenvector matrix is an eigenvector
				eigenvector = eigenvectors[:,i]
				# We are just looking for the complex conjugate eigenvectors - ignore the real one
				if (abs(eigenvector.imag) > .01).any():				
					# Convert to inhomogenous coordinates
					circularPoints.append(eigenvector[0:2]/eigenvector[2])
			
			# Use the homography to warp image1
			warp = cv2.warpPerspective(image1.copy(), M, (image2.shape[1], image2.shape[0]))
			
			# Format the output image
			output = np.zeros((matchImage.shape[0]+warp.shape[0], matchImage.shape[1], 3))
			# Draw the matches on the top
			output[0:matchImage.shape[0], 0:matchImage.shape[1]] = matchImage
			# Subtract the warped first image from the second image, draw the result (see how it matches up)
			output[matchImage.shape[0]:matchImage.shape[0]+warp.shape[0], 0:warp.shape[1]] = image2-warp
			# Draw the warped first image to compare against the second image
			output[matchImage.shape[0]:matchImage.shape[0]+warp.shape[0], warp.shape[1]:2*warp.shape[1]] = warp

			return [circularPoints, output, M]
			
		else:
			print("No valid homography found")
			return []

	else:
		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None
		return []	

# Build the linear conic fitting matrix from the input circular points
def getConic(points):
	M = []
	
	#  Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
	#  
	# | x1^2  x1y1  y1^2  x1  y1  1 |   | A | 
	# | x2^2  x2y2  y2^2  x2  y2  1 |   | B | 
	# | 			.				|   | C |  = 0
	# | 			.				|   | D | 
	# | 			.				|   | E | 
	# | 							|   | F |  
	# | xn^2  xnyn  yn^2  xn  yn  1 |   
	for point in points:
		M.append(np.array([point[0]**2, point[0]*point[1], point[1]**2, point[0], point[1], 1]))
	
	return np.vstack(M)	

# Build the linear conic fitting matrix from the input circular points - focal lengths only
def getFocalLengthConic(points):
	M = []
	
	#  Ax^2 + Cy^2 + F = 0
	#  
	# | x1^2  y1^2  1 |   | A | 
	# | x2^2  y2^2  1 |   | C |  = 0
	# | 	...		  |	  | F |  
	# | xn^2  yn^2  1 |   
	for point in points:
		M.append(np.array([point[0]**2, point[1]**2, 1]))
	
	return np.vstack(M)	

# Given the linear conic fitting matrix, solve for the camera matrix	
def getCameraParameters(A):
	
		# Use the SVD to fit the IAC
		u,s,v = npl.svd(A)

		print("u: ",u)
		print("s: ",s)
		print("v: ",v)

		# Get right singular vector (rows in numpy svd)
		# Numpy sorts the singular vectors by worst fit to best
		conic = v[-1,:]
		
		# Put the conic in matrix form
		a=conic[0]
		b=conic[1]
		c=conic[2]
		d=conic[3]
		e=conic[4]
		f=conic[5]
		iac = np.array([[a, b/2, d/2],[b/2, c, e/2],[d/2, e/2, f]])
		# and normaliza
		iac /= f
		
		print("iac: ",iac)

		# The dual of the IAC is it's inverse
		diac = npl.inv(iac)

		print("diac", diac)

		# DIAC = K*K^T, so use cholesky decomp
		chol = npl.cholesky(diac)
		K = chol.T.conj()
		
		# Normalize
		K /= K[2][2]

		return K

# Given the linear conic fitting matrix, solve for the camera matrix - focal length only	
def getFocalLength(A):

	# Use the SVD to fit the IAC
	u,s,v = npl.svd(A)

	# Get right singular vector (rows in numpy svd)
	# Numpy sorts the singular vectors by worst fit to best		
	conic = v[-1,:]	
	
	# Put the conic in matrix form (skew, ppt = 0)
	a=conic[0]
	c=conic[1]
	f=conic[2]
	iac = np.array([[a, 0, 0],[0, c, 0],[0, 0, f]])
	# and normalize
	iac /= f
	
	print("iac: ",iac)
	
	# The dual of the IAC is it's inverse
	diac = npl.inv(iac)

	print("diac", diac)
	
	# DIAC = K*K^T, so use cholesky decomp
	chol = npl.cholesky(diac)
	K = chol.T.conj()
	
	# Normalize
	K /= K[2][2]

	return K





