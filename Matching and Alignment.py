import cv2
import numpy as np

# Load images
img1 = cv2.imread('porsche1.png')
img2 = cv2.imread('porsche2.png')

img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect features using ORB
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Match features using BFMatcher with Hamming distance for ORB
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test to select good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw matches
match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

cv2.imshow("Matches", match_img)
cv2.imwrite('./Exp 04 - Matching and Alignment/matches.png', match_img)

cv2.waitKey(0)
cv2.destroyAllWindows()