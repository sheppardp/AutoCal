Planar Calibration Solver
Matt Kaplan CISC 849 - Autonomous Robot Vision

RUN:
This application was built on Xubuntu with Python3 using OpenCV 3.1, Tkinter, and  PIL/pillow.

	1.  Follow the PyImageSearch Instructions to install OpenCV 3.1 with Python 3.4:
http://www.pyimagesearch.com/2015/07/20/install-opencv-3-1-and-python-3-4-on-ubuntu/

	2.  Install Tkinter:  sudo apt-get install python3-tk

	3.  Install Tkinter/PIL:  sudo apt-get install python3-PIL

	4.  Install PIL/pillow:  pip install Pillow

Now you can run the program with “Python SolverGUI.py"

USAGE:
First, use "Number of Image Pairs" to set the number of image pairs to use.
Then, decide if you want to use the "Focal Length Only" checkbox to ignore camera skew and principal point offset.
You'll need at least one image pair for "Focal Length Only", and at least three otherwise.
You can overfit and use up to ten image pairs.
Make these decisions first, as they will both reset the loaded camera image pairs.

Next, use the "Get Image N" buttons to load the camera image pairs.
Once you have enough, click the "Run Calibration" button to run the algorithm.

Now, the feature matching and transformation of the first pair will show up in the display window.  The top of the image is the matched features, the bottom right is the first image transformed to match the second, and the bottom left is a diff of that transform against the input second image.  The resulting homography parameters and calculated circular points are displayed below.

You can scroll back and forth through the match images, homographies and circular points for each image pair with the "Previous Matches" and "Next Matches" buttons.  The calcuated camera calibration matrix solution is displayed at the bottom.

If the DIAC matrix is not positive definite, the algorithm cannot perform the Cholesky decomposition, and the bottom text will let you know.

However, if some of your image pair matches drop below the feature match threshold (6 matches) for calcuating a homography, and losing the corresponding pairs of circular point from the calculation drops you below the minimum points necessary to calculate the calibration parameters, the app will fail and let you know on the console.

The console also prints several of the intermeadiate results: eigenvectors, linear conic fitting matrix, SVD, IAC, DIAC.
