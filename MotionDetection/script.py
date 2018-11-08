
"""
Here we are detecting motion by saving last frame and comparing next 2 frames with itself.
This gives the difference in the frames and thus detects motion.

"""

import cv2, time
from datetime import datetime


# Take difference of 3 frames
def diffImg(t0, t1, t2):
    # absdiff - absolute difference between 2 frames
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)


# first_frame = None
video = cv2.VideoCapture(0)     # Connect to camera

status_list = [None, None]
times = []

# Read a frame, convert it to grayscale, blur it and store in t_minus
frame = video.read()[1]
t_minus = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
t_minus = cv2.GaussianBlur(t_minus, (21, 21), 0)

# Read another frame, convert it to grayscale, blur it and store in t
frame = video.read()[1]
t = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
t = cv2.GaussianBlur(t, (21, 21), 0)

# Read another frame, convert it to grayscale, blur it and store in t_plus
frame = video.read()[1]
t_plus = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # convert BGR image to GrayScale
t_plus = cv2.GaussianBlur(t_plus, (21, 21), 0)      # to blur image

# Record frames until user quits
while True:
    status = 0
    delta_frame = diffImg(t_minus, t, t_plus)

    # threshold - define threshold to change pixel color of frame.
    # @param delta_frame: image/frame in grayscale.
    # @param 90: threshold value of pixel, everything under this value is converted to 0 or black.
    # @param 255: max value of pixel which 255 for grayscale.
    # @param cv2.THRESH_BINARY: Type of threshold
    thresh_frame = cv2.threshold(delta_frame, 10, 255, cv2.THRESH_BINARY)[1]

    # Dilute the boundaries of object or make it thicker for detection
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Define countours
    (_,cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status=1

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    # cv2.imshow("First frame", first_frame)
    # cv2.imshow("Gray Frame", gray)
    # cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    t_minus = t
    t = t_plus

    # Read another frame
    check, frame = video.read()
    t_plus = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t_plus = cv2.GaussianBlur(t_plus, (21, 21), 0)

    # Wait for key to be pressed
    key = cv2.waitKey(1)
    # print(gray)
    # print(delta_frame)

    # If pressed key is 'q' then break the loop
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break


# print(status_list)
# print(times)

video.release()
cv2.destroyAllWindows()
