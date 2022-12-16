#Open CV2 and imutils Imports
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
#Tkinter -> GUI import
from tkinter import *

#Webcam Function to Grab Picture
def webcam(img_name):
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #captureDevice = camera
    cv2.namedWindow("Space to capture and ESC to exit")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab a picture")
            break
        cv2.imshow("Space to capture and ESC to exit", frame)
        k = cv2.waitKey(1)
        if k%256 == 27: #ESC Pressed
            print("Escape key was hit, closing webcam.")
            break
        elif k%256 == 32: #Space Pressed
            cv2.imwrite(img_name, frame)
            print("{} was written!".format(img_name))
    cam.release()
    cv2.destroyAllWindows()

#Define a midpoint function to quickly find midpoints for our objects
def midpoint(A, B):
    return (((A[0] + B[0]) * 0.5), (A[1] + B[1]) * 0.5)

#Function for finding contours and drawing object dimensions
def findDimensions(img_name, ref_width, ref_name):
    #Read in image and apply filters over to find contours
    image = cv2.imread(img_name)
    if image is None:
        error = "Error finding image, make sure you indicate the path to the image and extension (i.e. png, jpg, etc)"
        MyWindow(window).output(error)
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
    gray = cv2.GaussianBlur(gray, (7,7), 0) #create a blur on the grayscale image

    #Edge detection using canny, dilate, and erode:
    edge = cv2.Canny(gray, 50, 100) #Canny is edge detection -> 50 and 100 are thresholds
    edge = cv2.dilate(edge, None, iterations=1)
    edge = cv2.erode(edge, None, iterations=1)

    #Use image with edge detection to now find contours in an image
    #Define new contours variable that holds the given contours in an image
    cntour = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntour = imutils.grab_contours(cntour)
    (cntour, _) = contours.sort_contours(cntour)
    pixelsPerMetric = None #Defined once we find our reference object

    #Loop through our contours individually
    for i in cntour:
        #If contour area < 150 pixels, not sufficiently large to be an object
        if cv2.contourArea(i) < 150: 
            continue

        #create a copy of our image to draw dimensions on them
        imgCpy = image.copy() 

        #returns coordinates for the rectangle that fits in the contour, may or may not be rotated
        box = cv2.minAreaRect(i)

        #Compute the rotated bounding box of the image
        #This is to take into account different versions of CV2
        #cv2.cv.BoxPoints is for OpenCV 2.4
        #cv2.boxPoints is for OpenCV 3
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)

        #Put the coordinates for our rectangle in an integer array
        box = np.array(box, dtype="int") 

        #Reorder the points in the box to follow perspective orientation:
        #Top-left, top-right, bottom-right, the bottom-left orientation
        box = perspective.order_points(box)

        #Draw box using coordinates on the copied image - color is red
        cv2.drawContours(imgCpy, [box.astype("int")], -1, (255, 0, 0), 2)

        #Draw the points on the box around our object
        #Points are a green filled circle with a radius of 5 pixels
        for (x,y) in box:
            cv2.circle(imgCpy, (int(x), int(y)), 5, (0, 255, 0), -1)

        #Set variables for the vertices on our box
        (topL, topR, botR, botL) = box #set in perspective orientation from before

        #Get the midpoint of our vertices using midpoint function
        (topX, topY) = midpoint(topL, topR) #top
        (botX, botY) = midpoint(botL, botR) #bottom
        (leftX, leftY) = midpoint(topL, botL) #left
        (rightX, rightY) = midpoint(topR, botR) #right

        #Draw in the midpoints
        #Points are a blue filled circle with a radius of 5 pixels
        cv2.circle(imgCpy, (int(topX), int(topY)), 5, (0, 0, 255), -1) 
        cv2.circle(imgCpy, (int(botX), int(botY)), 5, (0, 0, 255), -1) 
        cv2.circle(imgCpy, (int(leftX), int(leftY)), 5, (0, 0, 255), -1) 
        cv2.circle(imgCpy, (int(rightX), int(rightY)), 5, (0, 0, 255), -1) 

        #Draw lines in between the midpoints
        #Lines are magenta with thickness of 2 pixels 
        cv2.line(imgCpy, (int(topX), int(topY)), (int(botX), int(botY)), (255, 0, 255), 2)
        cv2.line(imgCpy, (int(leftX), int(leftY)), (int(rightX), int(rightY)), (255, 0, 255), 2)

        #Find distance between points by using Euclidean distance function
        dVert = dist.euclidean((topX, topY), (botX, botY))
        dHoriz = dist.euclidean((leftX, leftY), (rightX, rightY))

        #Reference object must be the first contour - therefore must be the left most object
        #Compute pixelsPerMetric given a reference width for our first object
        #Only computed once
        if pixelsPerMetric is None:
            pixelsPerMetric = dHoriz / ref_width
            refTopX = topX
            refTopY = topY
            
        cv2.putText(imgCpy, "Reference Object: {}".format(ref_name), (int(refTopX - 45), int(refTopY - 30)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        #Compute the dimensions of the object
        objWidth = dVert / pixelsPerMetric
        objHeight = dHoriz / pixelsPerMetric

        #Draw in the dimensions
        cv2.putText(imgCpy, "{:.1f}in".format(objWidth), (int(topX - 15), int(topY - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(imgCpy, "{:.1f}in".format(objHeight), (int(rightX + 10), int(rightY)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        #Display image
        cv2.imshow("Image", imgCpy)
        cv2.waitKey(0)

#Define a class for a GUI
class MyWindow:
    global imageName
    global referenceWidth
    def __init__(self, window):
        global v0 #File system variable
        global v1 #Background remover variable
        v0 = IntVar()
        v0.set(0) # = 0 means default to using webcam as import
        v1 = IntVar()
        v1.set(0)
        #Define Buttons, Labels, and Text Entries
        self.r1 = Radiobutton(window, text="Import through webcam", variable = v0, value = 0)
        self.r2 = Radiobutton(window, text="Import through file system", variable = v0, value = 1)
        self.lbl1 = Label(window, text="Name of image: ")
        self.lbl2 = Label(window, text="Width of reference object(Inches): ")
        self.lbl3 = Label(window, text="Note: \nImporting Images: Specify the entire path\nWebcam: Specify extension (i.e. .png)")
        self.lbl4 = Label(window, text="Name of reference object: ")
        self.lbl5 = Label(window, text="Output: ")
        self.e1 = Entry() #Name/path of image
        self.e2 = Entry() #Width of reference in inches
        self.e3 = Entry() #Name of reference object to display
        self.e4 = Entry() #Empty entry box for outputting updates
        self.btn = Button(window, text="Start", command=self.start)
        #Set Place for Widgets
        self.r1.place(x=0, y=0) #Radiobutton: Webcam Option
        self.r2.place(x=0, y=20) #Radiobutton: Import Option
        self.lbl1.place(x=0, y=100) #Label: Name of image
        self.lbl2.place(x=0, y=130) #Label: Width of Reference
        self.lbl3.place(x=180, y=0) #Label: Warning/Note
        self.lbl4.place(x=0, y=155) #Label: Name of reference
        self.lbl5.place(x=0, y=240) #Label: Output
        self.e1.place(x=200, y=100) #Entry: Name of Image Entry
        self.e2.place(x=200, y=130) #Entry: Width of Reference Entry
        self.e3.place(x=200, y=155) #Entry: Name of reference
        self.e4.place(x=50, y=240, width=500, height=30) #Entry: Output
        self.btn.place(x=180, y=180) #Button to start
    #Call function to find dimensions
    def output(self, text):
        self.e4.delete(0, 'end')
        self.e4.insert(END, text)
    def start(self):
        self.e4.delete(0, 'end') 
        imageName = str(self.e1.get())
        referenceWidth = float(self.e2.get())
        referenceName = str(self.e3.get())
        #Use webcam to take picture
        if v0.get() == 0:
            #Take a picture to use
            if "." in imageName:
                webcam(imageName)
                findDimensions(imageName, referenceWidth, referenceName)
                self.output("Dimensions found using webcam")
            else:
                error = "Remember to include file extension in the image file name (i.e. png, jgp, etc)"
                self.output(error)
        #Use an imported picture - must specify entire path of image
        else: 
            findDimensions(imageName, referenceWidth, referenceName)
            self.output("Dimensions found using imported image")

window = Tk()
mywin = MyWindow(window)
window.title('Find Dimensions In Picture')
window.geometry("600x400+10+20")
window.mainloop()