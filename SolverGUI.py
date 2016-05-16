from tkinter import *
from tkinter import filedialog
import Calibrate
import cv2
import numpy as np
from numpy import linalg as npl
from OpenCVCanvas import OpenCVCanvas
from ImagePairWidget import ImagePairWidget

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
		
		halfWidth = str(int(screen_width/2))
		halfHeight = str(int(screen_height/2)+200)
		
		# put the window at the top midlle of the screen
		self.geometry(halfWidth+"x"+halfHeight+"+"+halfWidth+"+0")
		
		# Build the menu bars
		self.imagePairWidgetMenu = Frame(self)
		self.imagePairWidgetMenu.pack()
		#menu2 = Frame(self)
		#menu2.pack()
		#menu3 = Frame(self)
		#menu3.pack()
		menu4 = Frame(self)
		menu4.pack()
		
		imageRow = Frame(self)
		imageRow.pack()
		
		menu5 = Frame(self)
		menu5.pack()
		menu6 = Frame(self)
		menu6.pack()
		menu7 = Frame(self)
		menu7.pack()
		
		self.images = {}
		self.pairLabels = {}
		
		# Allow the user to change the number of steps in the orbit
		self.numImagePairs = StringVar()
		self.numImagePairs.set(3)
		self.imagePairWidgetList = []
		self.setNumImagePairs(int(self.numImagePairs.get()))
		Label(menu4, text = "Number of Image Pairs").pack(side=LEFT)
		self.numImagePairsSpinbox = Spinbox(menu4, from_=3, to=10, increment=1, textvariable=self.numImagePairs, command=lambda: self.setNumImagePairs(int(self.numImagePairs.get())))
		self.numImagePairsSpinbox.pack(side=LEFT)
		
		self.calibrateButton = Button(menu4, text="Run Calibration", state='disabled', command = self.calibrate)
		self.calibrateButton.pack(side=LEFT)
		
		self.imageCanvas1 = OpenCVCanvas(imageRow, height=windowHeight, width=windowWidth)
		self.imageCanvas1.pack(side=LEFT)
		
		self.homographyLabel = Label(menu5)
		self.homographyLabel.pack(side=LEFT)
		
		self.circularPointLabel = Label(menu6)
		self.circularPointLabel.pack(side=LEFT)
		
		self.pvsMatchImageButton = Button(menu7, text="Previous Matches", state='disabled', command = lambda: self.setMatchImage(self.currentMatchImage-1))
		self.pvsMatchImageButton.pack(side=LEFT)
		self.nextMatchImageButton = Button(menu7, text="Next Matches", state='disabled', command = lambda: self.setMatchImage(self.currentMatchImage+1))
		self.nextMatchImageButton.pack(side=LEFT)
		self.currentMatchImage = 0		
		self.matchImages = []
				
	def setNumImagePairs(self, numImagePairs):
		print(numImagePairs)
		
		for imagePairWidget in self.imagePairWidgetList:
			imagePairWidget.destroy()
		
		self.imagePairWidgetList = []
		for i in range(numImagePairs):
			imagePairWidget = ImagePairWidget(self.imagePairWidgetMenu, i*2,self.images, self.pairLabels, self.pairLoaded)
			imagePairWidget.pack()
			self.imagePairWidgetList.append(imagePairWidget)
			
	def	pairLoaded(self):
		if len(self.images) > 5:
			self.calibrateButton.config(state='normal')
					
	def loadImage(self, label):
		openImage = filedialog.askopenfilename()
		
		if openImage:			
			image = cv2.imread(openImage)
			if image is not None:
				# OpenCV reads in BGR - switch to RGB
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				self.images[label]=[image, openImage.split('/')[-1]]
				
				currentLabel = self.labelPairs[int(label/2)]
				
				try:
					firstImage = self.images[int(label/2)*2][1]
				except KeyError:
					firstImage = ""
				
				try:
					secondImage = self.images[int(label/2)*2+1][1]
				except KeyError:
					secondImage = ""
					
				currentLabel.config(text="Image Pair Set to "+firstImage+", "+secondImage)
				
				if len(self.images) > 5:
					self.calibrateButton.config(state='normal')
					
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
			
			
			homographyString = ("Homography: | {:.5f}   {:.5f}   {:.5f} |\n"+
							    "            | {:.5f}   {:.5f}   {:.5f} |\n"+
							    "            | {:.5f}   {:.5f}   {:.5f} |").format(self.matchImages[pos][2][0][0], self.matchImages[pos][2][0][1], self.matchImages[pos][2][0][2], 
																				   self.matchImages[pos][2][1][0], self.matchImages[pos][2][1][1], self.matchImages[pos][2][1][2], 
																				   self.matchImages[pos][2][2][0], self.matchImages[pos][2][2][1], self.matchImages[pos][2][2][2])
			self.homographyLabel.config(text=str(homographyString))
			
			if pos < len(self.matchImages) - 1:
				self.nextMatchImageButton.config(state='normal')
			else:
				self.nextMatchImageButton.config(state='disabled')
			if pos  > 0:
				self.pvsMatchImageButton.config(state='normal')
			else:
				self.pvsMatchImageButton.config(state='disabled')
			self.currentMatchImage = pos
										
	def calibrate(self):
		
		np.set_printoptions(precision=2, suppress=True, linewidth=200)

		self.matchImages = []
		for i in range(int(len(self.images)/2)):
			self.matchImages.append(Calibrate.getCriticalPoints(self.images[i*2][0], self.images[i*2+1][0]))
		
		#critical0, matchImage0, homography0 = Calibrate.getCriticalPoints(self.images[0][0], self.images[1][0])
		#critical1, matchImage1, homography1 = Calibrate.getCriticalPoints(self.images[2][0], self.images[3][0])
		#critical2, matchImage2, homography2 = Calibrate.getCriticalPoints(self.images[4][0], self.images[5][0])
		
		#self.matchImages = [[matchImage0, critical0, homography0], [matchImage1, critical1, homography1], [matchImage2, critical2, homography2]]
		self.setMatchImage(0)

		#A = np.array(Calibrate.getFocalLengthConic(critical1))

		#A = np.array(Calibrate.getConic(critical0+critical1+critical2))

		DLT = []
		for matchImage in self.matchImages:
			DLT +=  matchImage[0]

		A = np.array(Calibrate.getConic(DLT))

		print("A: ", A)
		
		K = Calibrate.getCameraParameters(A)
		#K = Calibrate.getFocalLength(A)

		print("K:", K)


app = SolverGUI()
app.mainloop()
