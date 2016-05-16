from tkinter import *
import cv2

class ImagePairWidget(Frame):
			  
	def __init__(self, parent, m, imageList, labelPairList, pairLoadedCallback):
		Frame.__init__(self, parent)  

		self.m = m

		Button(self, text="Get Image "+str(m+1), command = lambda: self.loadImage(m)).pack(side=LEFT)
		Button(self, text="Get Image "+str(m+2), command = lambda: self.loadImage(m+1)).pack(side=LEFT)
		self.pair1Label = Label(self)
		self.pair1Label.pack(side=LEFT)
		
		self.images = imageList
		self.labelPairs = labelPairList
		
		labelPairList[int(m/2)] = self.pair1Label
		
		self.pairLoadedCallback = pairLoadedCallback

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
				
				self.pairLoadedCallback()

