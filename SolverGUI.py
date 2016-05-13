from tkinter import *
from tkinter import filedialog
import Calibrate
import cv2
import numpy as np
from numpy import linalg as npl
from OpenCVCanvas import OpenCVCanvas

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
		menu1 = Frame(self)
		menu1.pack()
		menu2 = Frame(self)
		menu2.pack()
		menu3 = Frame(self)
		menu3.pack()
		menu4 = Frame(self)
		menu4.pack()
		
		imageRow = Frame(self)
		imageRow.pack()
		
		menu5 = Frame(self)
		menu5.pack()
		menu6 = Frame(self)
		menu6.pack()
		
		self.images = {}
		
		Button(menu1, text="Get Image 1", command = lambda: self.loadImage(0)).pack(side=LEFT)
		Button(menu1, text="Get Image 2", command = lambda: self.loadImage(1)).pack(side=LEFT)
		self.pair1Label = Label(menu1)
		self.pair1Label.pack(side=LEFT)
		
		Button(menu2, text="Get Image 3", command = lambda: self.loadImage(2)).pack(side=LEFT)
		Button(menu2, text="Get Image 4", command = lambda: self.loadImage(3)).pack(side=LEFT)
		self.pair2Label = Label(menu2)
		self.pair2Label.pack(side=LEFT)
		
		Button(menu3, text="Get Image 5", command = lambda: self.loadImage(4)).pack(side=LEFT)
		Button(menu3, text="Get Image 6", command = lambda: self.loadImage(5)).pack(side=LEFT)
		self.pair3Label = Label(menu3)
		self.pair3Label.pack(side=LEFT)
		
		self.pairLabels = [self.pair1Label, self.pair2Label, self.pair3Label]
		
		self.calibrateButton = Button(menu4, text="Run Calibration", state='disabled', command = self.calibrate)
		self.calibrateButton.pack(side=LEFT)
		
		self.imageCanvas1 = OpenCVCanvas(imageRow, height=windowHeight, width=windowWidth)
		self.imageCanvas1.pack(side=LEFT)
		
		self.criticalPointLabel = Label(menu5)
		self.criticalPointLabel.pack(side=LEFT)
		
		self.pvsMatchImageButton = Button(menu6, text="Previous Matches", state='disabled', command = lambda: self.setMatchImage(self.currentMatchImage-1))
		self.pvsMatchImageButton.pack(side=LEFT)
		self.nextMatchImageButton = Button(menu6, text="Next Matches", state='disabled', command = lambda: self.setMatchImage(self.currentMatchImage+1))
		self.nextMatchImageButton.pack(side=LEFT)
		self.currentMatchImage = 0		
		self.matchImages = []
		

		
		self.labelPairs = {0:self.pair1Label, 1:self.pair2Label, 2:self.pair3Label}				
		
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
			self.imageCanvas1.publishArray(self.matchImages[pos][0])
			
			for i in range(len(self.pairLabels)):
				if i == pos:
					self.pairLabels[i].configure(background='yellow')
				else:
					self.pairLabels[i].configure(background='light gray')
			
			critString = "Circular Points: ({:.5f}, {:.5f})   ({:.5f}, {:.5f})".format(self.matchImages[pos][1][0][0], self.matchImages[pos][1][0][1], self.matchImages[pos][1][1][0], self.matchImages[pos][1][1][1])
			
			self.criticalPointLabel.config(text=str(critString))
			
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

		
		critical0, matchImage0 = Calibrate.getCriticalPoints(self.images[0][0], self.images[1][0])
		critical1, matchImage1 = Calibrate.getCriticalPoints(self.images[2][0], self.images[3][0])
		critical2, matchImage2 = Calibrate.getCriticalPoints(self.images[4][0], self.images[5][0])
		
		self.matchImages = [[matchImage0, critical0], [matchImage1, critical1], [matchImage2, critical2]]
		self.setMatchImage(0)


		A = np.array(Calibrate.getConic(critical0+critical1+[critical2[0]]))

		print("A: ", A)

		u,s,v = npl.svd(A)

		print(u,s,v)

		conic = v[-1,:]

		a=conic[0]
		b=conic[1]
		c=conic[2]
		d=conic[3]
		e=conic[4]
		f=conic[5]

		iac = np.array([[a, b/2, d/2],[b/2, c, e/2],[d/2, e/2, f]])
		
		print("iac: ",iac)

		diac = npl.inv(iac)

		print(diac)

		K = npl.cholesky(diac)

		print(K)


app = SolverGUI()
app.mainloop()
