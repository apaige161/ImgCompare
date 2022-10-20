from asyncio.windows_events import NULL
import cv2
from cv2 import dilate
import imutils
import numpy as np
import time
import sys
from sys import argv


start_time = time.time()





title1 = 'Original'
title2 = 'Altered'
if(len(sys.argv) == 2):
    path1 = argv[1]
    path2 = argv[2]
else:
    path1 = 'img/beach.jpeg'
    path2 = 'img/beach_altered.jpeg'

savedImgPath = "imgDiff/newImg.jpeg"

# Get data from user
print('\n\nComparing 2 images...\n\n')


# Load 2 images
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

if (img1.any() or img2.any()) is None:
    sys.exit("Could not read the image.")

# resize
imgX = 600
imgY = 360
img1 = cv2.resize(img1, (imgX,imgY))
img2 = cv2.resize(img2, (imgX,imgY))

# greyscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# find differences between two images
diff = cv2.absdiff(gray1, gray2)
# cv2.imshow("diff(img1,img2)", diff)

# TODO: further investigation on how this part works
    # GOAL: reduce processing time be not running other functions if not needed
if not np.array_equiv(gray1, gray2):
    print("Difference found at gray scale, processing...")

else:
    print("No difference between images found at gray scale")

# apply threshold
# get binary image out of grayscale
# cv2.threshold(image, pixel_value_threshold, max_pixel_values, type_of_threshold)
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # finds optimal threshold value which is returned as the first output
# cv2.imshow("Threshold", thresh)

# dilation
# draw new shape around differences -> then increase size
kernel = np.ones((5,5), np.uint8)
dilate = cv2.dilate(thresh, kernel, iterations=2)
# cv2.imshow("Dilation", dilate)

# find contours
contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

if not contours:

    print("No difference found at contour check")
else: 
    print("Difference found at contour check, processing...")
    # loop over contours, draw rectangle around it
    for contour in contours:
        if(cv2.contourArea(contour) > 50):
            # calc bounds
            x, y, w, h = cv2.boundingRect(contour)
            # draw rectangle 
            cv2.rectangle(img1, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.rectangle(img2, (x,y), (x+w, y+h), (0,0,255), 2)
        else:
            print("No difference")

    # show final images with differences
    # stack images
    x = np.zeros( (360,10,3), np.uint8) # add space between imgs
    result = np.hstack( (img1, x, img2) )

    # create a border
    result = cv2.copyMakeBorder(result,10,35,10,10,cv2.BORDER_CONSTANT)

    # write text on image
    numbDifferences = str(len(contours))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,'Number of differences: '+ numbDifferences,(10,imgY - 10), font, 2,(255,255,255),4,cv2.LINE_AA)

    #save image
    cv2.imwrite(savedImgPath, result)
    print("\n\nImage saved at '" + savedImgPath + "'\n\n")


cv2.waitKey(0)
cv2.destroyAllWindows()
print("--- %s seconds ---" % (time.time() - start_time))