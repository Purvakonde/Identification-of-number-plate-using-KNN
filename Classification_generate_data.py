import sys
import numpy as np
import cv2
import os

contour_area=100
resize_img_width=20
resize_img_height=30

def main():
    img=cv2.imread("training_chars.png")
    if img is None:
        print("Cannot Read from image")
        os.system("Pause")
        return
    
    #Convert image to grayscale
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #Gaussian Blur 
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),0)
    
    #thresholding
    imgThresh=cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    imgThreshCopy=imgThresh.copy()      #making a copy so that it can be modified
    contours,hierarchy=cv2.findContours(imgThreshCopy,
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,255,0), 1)
    flattened_images= np.empty((0,resize_img_width*resize_img_height))
    classifications=[]
    
    chars=[ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
           ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
           ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
           ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]
    
    for contour in contours:
        if cv2.contourArea(contour)>contour_area:
            [X,Y,W,H]=cv2.boundingRect(contour)
            
            cv2.rectangle(img, (X,Y), (X+W,Y+H), (255,0,0),2)
            img_roi=imgThresh[Y:Y+H, X:X+W]
            img_roi_resize=cv2.resize(img_roi,(resize_img_width,resize_img_height))
            
        cv2.imshow("Region of Interest",img_roi)
        cv2.imshow("Resized Region of Interest",img_roi_resize)
        cv2.imshow("Training Data",img)
        char=cv2.waitKey(0)
        if char==27:
            sys.exit()
        elif char in chars:
            classifications.append(char)
            
            #convert multi-dimensional arrays into a 1-D array and feed the information to the classification model.
            flattened_image=img_roi_resize.reshape((1,resize_img_width*resize_img_height))
            flattened_images=np.append(flattened_images,flattened_image,0)
    
    float_classifications=np.array(classifications,np.float32)
    classifications=float_classifications.reshape((float_classifications.size,1))
    print("Successful !!")
    
    np.savetxt("Classifications.txt", classifications)
    np.savetxt("Flattened_Images.txt",flattened_images)
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()