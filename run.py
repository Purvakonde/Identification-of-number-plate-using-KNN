#Identify/crop no. plate from the whole image

import cv2
import imutils
import numpy as np
import operator
import os

min_contour_area=100
resize_img_width=20
resize_img_height=30

######## Getting the region of interest i.e. no.plate
def identifyPlate(img):
       # img= cv2.imread('Car Images/5.jpeg')
        img=imutils.resize(img,width=500)
        
        cv2.imshow("Original Car Image",img) 
        cv2.waitKey(0)
        
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grayscale Image",img_gray) 
        cv2.waitKey(0)
        
        #Gaussian Blur 
        img_blur=cv2.GaussianBlur(img_gray,(5,5),0)
        cv2.imshow("Gaussian Blur",img_gray) 
        cv2.waitKey(0)
        
        #Edge identification
        img_edge = cv2.Canny(img_blur, 170, 200)
        cv2.imshow("Edge Detection", img_edge)
        cv2.waitKey(0)
        img_edge_copy=img_edge.copy()   
        contours,hierarchy=cv2.findContours(img_edge_copy,
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        
        #copy of original image
        img_copy1=img.copy()
        cv2.drawContours(img_copy1, contours, -1, (0,255,0), 1)
        cv2.imshow("All Contours",img_copy1)
        cv2.waitKey(0)
        
        #sorting contours based on their area, minimum area=30
        contours=sorted(contours,key=cv2.contourArea, reverse=True)[:30]
        number_plate_contour= None
        
        #Displaying top 30 contours on the image
        img_copy2=img.copy()
        cv2.drawContours(img_copy2, contours, -1, (0,255,0), 1)
        #cv2.imshow("30 Contours",img_copy2)
        #cv2.waitKey(0)
        
        index=1  #Store the cropped image in certain location
        for c in contours:
            perimeter=cv2.arcLength(c,True)  #perimeter of each contour
            approx=cv2.approxPolyDP(c,0.02*perimeter,True) #approximate the shape of polygonal curves and returns no. of edges for each contour
            
            if len(approx)==4:   #as we need rectangle which has 4 sides
                number_plate_contour=approx
                
                X,Y,W,H=cv2.boundingRect(c)
                a=6
                new_img= img_gray[Y+a:Y+H-a, X+a:X+W-a]
                cv2.imwrite('Cropped Images/'+str(index+1)+'.jpg',new_img)
                index+=1
                break
            
        cv2.drawContours(img,[number_plate_contour], -1, (255,0,0), 1)
        cv2.imshow("Contour of number plate",img) 
        cv2.waitKey(0)
        
        Cropped_img_loc = 'Cropped Images/2.jpg'
        cv2.imshow("Cropped Image ", cv2.imread(Cropped_img_loc))
        cv2.waitKey(0) 
        TrainTest()
    
######### Train and test 
class ContourWithData():
    contour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < min_contour_area: return False        # much better validity checking would be necessary
        return True


def TrainTest():
    all_contours=[]
    valid_contours=[]
    try:
        classifications=np.loadtxt("Classifications.txt",np.float32) #labels
    except:
        print("Unable to open file")
        os.system("Pause")
        return
    
    try:
        flattened_img=np.loadtxt("Flattened_Images.txt",np.float32) #data points
    except:
        print("Unable to open file")
        os.system("Pause")
        return
    
    classifications=classifications.reshape((classifications.size,1))  #reshaping it to 1D np array
    
    knn=cv2.ml.KNearest_create()
    knn.train(flattened_img,cv2.ml.ROW_SAMPLE,classifications)  #datapoints, occupy row of samples from samples, labels
    #pickle.dump(knn,open('model.pkl','wb'))
    #model=pickle.load(open('model.pkl','rb'))
    
    test_img=cv2.imread('Cropped Images/2.jpg') # performing prediction on test image
    if test_img is None:
        print("Cannot read image")
        os.system("Pause")
        return
    
    #Convert image to grayscale
    imgGray=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    
    #Gaussian Blur 
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),0)
    
    #thresholding
    imgThresh=cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imshow("Threshold img",imgThresh)
    cv2.waitKey(0)
    imgThreshCopy=imgThresh.copy()      #making a copy so that it can be modified
    contours,hierarchy=cv2.findContours(imgThreshCopy,
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.contour = contour                                         # assign contour to contour with data
        #print(contourWithData.contour)
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.contour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.contour)           # calculate the contour area
        all_contours.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for

    for contourWithData in all_contours:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            valid_contours.append(contourWithData)       # if so, append to valid contour list
        # end if
    # end for

    valid_contours.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right
    
    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    for contourWithData in valid_contours:            # for each contour
        #if (contourWithData.intRectY[0] % contourWithData.intRectY==0):                                   # draw a green rect around the current char
                
            cv2.rectangle(test_img,                                        # draw rectangle on original testing image
                          (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                          (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                          (0, 255, 0),              # green
                          2)                        # thickness
    
            imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                               contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
    
            imgROIResized = cv2.resize(imgROI, (resize_img_width, resize_img_height))             # resize image, this will be more consistent for recognition and storage
    
            npaROIResized = imgROIResized.reshape((1, resize_img_width * resize_img_height))      # flatten image into 1d numpy array
    
            npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats
    
            retval, npaResults, neigh_resp, dists = knn.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest
    
            strCurrentChar = str(chr(int(npaResults[0][0])))   
    
            # print the results obtained
            print("Character:", strCurrentChar)
            print( "Label of the unknown data - ", npaResults )
            print( "Nearest neighbors -  ", neigh_resp )
            print( "Distance of each neighbor - ", dists )
            print("\n")                                          # get character from results
            
            strFinalString = strFinalString + strCurrentChar            # append current char to full string
            if(len(strFinalString)>10):
                strFinalString=strFinalString[1:11]
        
     
    print("Car Number: "+strFinalString+"\n")
    cv2.imshow("Car number plate",test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return strFinalString
img= cv2.imread('Test Images/113.jpg')  
identifyPlate(img)
