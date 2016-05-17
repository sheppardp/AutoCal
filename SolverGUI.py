from tkinter import *
from tkinter import filedialog
import Calibrate
import cv2
import numpy as np
from numpy import linalg as npl
from OpenCVCanvas import OpenCVCanvas
from ImagePairWidget import ImagePairWidget

# GUI wrapper aroung the calibration algorihtm
class SolverGUI(Tk):
	
	def __init__(self):
		# Call super constructor
		Tk.__init__(self)

		# Get the current screen dimensions...
		screen_width = self.winfo_screenwidth()
		screen_height = self.winfo_screenheight()
		# ... and size the display windows appropriately
		windowWidth = screen_width / 3
		windowHeight = (screen_height-200)/ 2
		
		# put the window at the top midlle of the screen
		halfWidth = str(int(screen_width/2))
		halfHeight = str(int(screen_height/2)+300)		
		self.geometry(halfWidth+"x"+halfHeight+"+"+halfWidth+"+0")
		
		# Build the widget rows
		# Image Pair selection
		self.imagePairWidgetMenu = Frame(self)
		self.imagePairWidgetMenu.pack()
		# Add'l input parameters
		menu4 = Frame(self)
		menu4.pack()		
		# Image display
		imageRow = Frame(self)
		imageRow.pack()
		# Homography display		
		menu5 = Frame(self)
		menu5.pack()
		# Circular Point display
		menu6 = Frame(self)
		menu6.pack()
		# Scroll image pair homographies buttons
		menu7 = Frame(self)
		menu7.pack()
		# Output calibration matrix
		menu8 = Frame(self)
		menu8.pack()
		
		# Allow the user to adjust the number of input image pairs
		self.numImagePairs = StringVar()
		self.numImagePairs.set(3)
		self.imagePairWidgetList = []
		Label(menu4, text = "Number of Image Pairs").pack(side=LEFT)
		self.numImagePairsSpinbox = Spinbox(menu4, from_=1, to=10, increment=1, textvariable=self.numImagePairs, command=lambda: self.setNumImagePairs(int(self.numImagePairs.get())))
		self.numImagePairsSpinbox.pack(side=LEFT)

		# Fire the calibration
		self.calibrateButton = Button(menu4, text="Run Calibration", state='disabled', command = self.calibrate)
		self.calibrateButton.pack(side=LEFT)
		
		# Allow the user to solely discover the focal lengths (no skew, (0,0) principal point)
		self.focalOnly = IntVar()
		self.focalOnly.set(0)
		focalOnlyCheckbutton = Checkbutton(menu4, text="Focal Length Only", variable=self.focalOnly, command=lambda: self.setFocalOnly(self.focalOnly.get())).pack(side=LEFT)
		
		# Show some output
		self.imageCanvas1 = OpenCVCanvas(imageRow, height=windowHeight, width=windowWidth)
		self.imageCanvas1.pack(side=LEFT)
		
		# Described each resolved image pair
		self.homographyLabel = Label(menu5)
		self.homographyLabel.pack(side=LEFT)		
		self.circularPointLabel = Label(menu6)
		self.circularPointLabel.pack(side=LEFT)
		
		# Allow the user to cycle through displays of the image pairs
		self.pvsMatchImageButton = Button(menu7, text="Previous Matches", state='disabled', command = lambda: self.setMatchImage(self.currentMatchImage-1))
		self.pvsMatchImageButton.pack(side=LEFT)
		self.nextMatchImageButton = Button(menu7, text="Next Matches", state='disabled', command = lambda: self.setMatchImage(self.currentMatchImage+1))
		self.nextMatchImageButton.pack(side=LEFT)
		
		# Show the calibration result
		self.calibrationMatrixLabel = Label(menu8)
		self.calibrationMatrixLabel.pack(side=LEFT)
		
		# Store the images, indexed by image number
		self.images = {}
		# Store access to the image pair labels
		self.pairLabels = {}
		# Set up the initial state of the image pairing
		self.setNumImagePairs(int(self.numImagePairs.get()))

		# Store the composite image matching outputs
		self.matchImages = []
		self.currentMatchImage = 0				
	
	# Switch between full calibration (5 d.o.f) and focal lengths only (2 d.o.f) modes
	def setFocalOnly(self, focalOnly):
		self.clearImagePairs()
		# In focal lengths only mode, we only need one image pair (we can use more)
		if focalOnly == 1:
			self.numImagePairs.set(1)
		# In full calibration matrix mode, we need at least 3 image pairs
		else:
			self.numImagePairs.set(3)
		self.setNumImagePairs(int(self.numImagePairs.get()))

	# For simplicity, any time we change the calibration mode or decide to use more image pairs, reset all currently loaded image pairs
	def clearImagePairs(self):
		self.calibrateButton.config(state='disabled')
		for imagePairWidget in self.imagePairWidgetList:
			imagePairWidget.destroy()
		
		self.imagePairWidgetList = []		
				
	# Change the number of input image pairs
	def setNumImagePairs(self, numImagePairs):
		# reset the pairs
		self.clearImagePairs()
		self.pairLabels = {}
		# and then build them back up
		for i in range(numImagePairs):
			imagePairWidget = ImagePairWidget(self.imagePairWidgetMenu, i*2,self.images, self.pairLabels, self.pairLoaded)
			imagePairWidget.pack()
			self.imagePairWidgetList.append(imagePairWidget)
		
	# depending on the calibration mode, when the user has input enough image pairs, allow use of the "Calibrate" button	
	def	pairLoaded(self):
		if len(self.images) > 5 or (self.focalOnly.get() == 1 and len(self.images) > 1):
			self.calibrateButton.config(state='normal')
		
	# Display the homography details for an image pair
	def setMatchImage(self, pos):
		if self.matchImages[pos] is not None:
			self.imageCanvas1.publishArray(self.matchImages[pos][1])
			
			for i in range(len(self.pairLabels)):
				if i == pos:
					self.pairLabels[i].configure(background='yellow')
				else:
					self.pairLabels[i].configure(background='light gray')
			
			circString = "Circular Points: ({:.5f}, {:.5f})   ({:.5f}, {:.5f})".format(self.matchImages[pos][0][0][0], self.matchImages[pos][0][0][1], self.matchImages[pos][0][1][0], self.matchImages[pos][0][1][1])
			
			self.circularPointLabel.config(text=str(circString))
			
			self.homographyLabel.config(text=self.matrixDisplayString(self.matchImages[pos][2], "Homography"))
			
			# Set up the pair result cycle buttons
			if pos < len(self.matchImages) - 1:
				self.nextMatchImageButton.config(state='normal')
			else:
				self.nextMatchImageButton.config(state='disabled')
			if pos  > 0:
				self.pvsMatchImageButton.config(state='normal')
			else:
				self.pvsMatchImageButton.config(state='disabled')
			self.currentMatchImage = pos
										
	# Run the calibration
	def calibrate(self):
		
		np.set_printoptions(precision=2, suppress=True, linewidth=200)

		self.matchImages = []
		
		# for each image pair...
		for i in range(int(len(self.images)/2)):
			# Find it's circular points (and output details)
			self.matchImages.append(Calibrate.getCircularPoints(self.images[i*2][0], self.images[i*2+1][0]))
		
		self.setMatchImage(0)

		# Prepare a matrix for linear fitting of conic parameters
		DLT = []
		for matchImage in self.matchImages:
			DLT +=  matchImage[0]

		try:
			# Find the focal lengths
			if self.focalOnly.get() == 1:			
				# Linear fit of IAC parameters
				A = np.array(Calibrate.getFocalLengthConic(DLT))
				print("A: ", A)
				# Decompose the conic to a calibration matrix
				K = Calibrate.getFocalLength(A)
			# Same with full calibration matrix
			else:
				A = np.array(Calibrate.getConic(DLT))
				print("A: ", A)
				K = Calibrate.getCameraParameters(A)

			self.calibrationMatrixLabel.config(text=self.matrixDisplayString(K, "K"))

		except np.linalg.linalg.LinAlgError:
			self.calibrationMatrixLabel.config(text="DIAC not positive definite")

	# Format a matrix for display
	def matrixDisplayString(self, M, label):
		matrixString = (label+": | {:.5f}   {:.5f}   {:.5f} |\n"+
			   ' '*len(label)+"  | {:.5f}   {:.5f}   {:.5f} |\n"+
			   ' '*len(label)+"  | {:.5f}   {:.5f}   {:.5f} |").format(M[0][0], M[0][1], M[0][2], 
																	   M[1][0], M[1][1], M[1][2], 
																	   M[2][0], M[2][1], M[2][2])
		return matrixString

app = SolverGUI()
app.mainloop()
