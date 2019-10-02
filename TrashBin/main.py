# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import json
import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# Instantiates a client
client = vision.ImageAnnotatorClient()


# The name of the image file to annotate
file_name = "image.jpg"


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

whitelist = json.loads(open("whitelist.json", "r").read())["whitelist"]

def recognize_image():
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print('Labels:')
    for label in labels:
        if label.score < 0.6:
            continue
        if label.description in whitelist:
            print(label.description, "in whitelist")
        else:
            print(label.description, "not in whitelist")
    
        print(label)

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])
 
# initialize the first frame in the video stream
firstFrame = None

frameCnt = 0

# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    text = "Unoccupied"
 
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break
 
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
        # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours

    flag = False

    maxArea = 0
    for c in cnts:
        # if the contour is too small, ignore it
        area = cv2.contourArea(c)
        if area < args["min_area"]:
            continue


 
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)

        if area > maxArea:
            maxArea = area
            maxX = x
            maxY = y
            maxW = w
            maxH = h
        
        frameCnt += 1
        flag = True
    
    if not flag:
        frameCnt = 0
    elif frameCnt > 60:
        frame2 = frame[maxY:(maxY + maxH), maxX:(maxX + maxW)]
        cv2.imshow("Security Feed", frame2)
        #time.sleep(2)
        cv2.imwrite(file_name, frame2)
        recognize_image()
        break

        
        

    # show the frame and record if the user presses a key
    #cv2.imshow("Security Feed", frame)
    #cv2.imshow("Thresh", thresh)
    #cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

