import cv2
import imutils
import numpy as np

title1 = 'Original'
path1 = 'img/beach.jpeg'
title2 = 'Altered'
path2 = 'img/beach_altered.jpeg'

# Get data from user
print('Comparing 2 images...')
# print('Enter path to original image:')
# path1 = input()
# print('Enter path to next image:')
# path2 = input()

# TODO: File found validation
# TODO: File type validation


# Load 2 images
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

# resize
img1 = cv2.resize(img1, (600,360))
img2 = cv2.resize(img2, (600,360))

# greyscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# find differences between two images
diff = cv2.absdiff(gray1, gray2)
cv2.imshow("diff(img1,img2)", diff)



# show the two images
cv2.imshow(title1, img1)
cv2.imshow(title2, img2)

cv2.waitKey(0)
cv2.destroyAllWindows()