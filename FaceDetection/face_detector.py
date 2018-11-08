# Face detection using OpenCV and haarcascade

import cv2
import glob


# Detect objects
def detect_objects(f_cascade, image, scaleFactor, minNeighbor, resizeFactor):
    gray_img = convertToGray(image)

    # scaleFactor - reduce size of image by what factor considering orignal image as base of pyramid
    #       and end point is top of it.
    # 1.05 means reduce by 5%
    faces = f_cascade.detectMultiScale(gray_img,
    scaleFactor = scaleFactor,
    minNeighbors = minNeighbor)

    # Its a numpy ndarray
    print(type(faces))
    # Have 4 values
    # coordinate of start point (x, y)
    # width and height of face detected
    print(faces)

    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    # Rectange - draws rectangle on image.
    # @params: <img-name>, <start-coordinate>, <end-coordinate>, <brg-color>, <width>
    # <end-coordinate> - diagonally opposite point from start point
    # <width> - width of border of rectangle
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

    if resizeFactor != 1:
        image = doResize(resizeFactor)

    return image


# Covert BGR image to Grayscale
def convertToGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Resize image
def doResize(factor):
    # Resize image to fit better
    # img.shape[1] - width of image
    # img.shape[0] - height of image
    # keeping the scale and reducing to 1/3 of original size
    resized = cv2.resize(img, (int(img.shape[1]/factor), int(img.shape[0]/factor)))
    return resized


## Cascades

# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
# faceProfile_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
# eye_cascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")



## To read all images
# images=glob.glob("*.jpeg")
#
# for image in images:
#     img = cv2.imread(image)
#     img = detect_objects(face_cascade, img, 1.05, 5, 1)
#     cv2.imshow("Gray", img)
#     cv2.waitKey(2000)
#     cv2.destroyAllWindows()


## To read individual images
img = cv2.imread("img9.jpeg")

img = detect_objects(face_cascade, img, 1.05, 5, 1)

cv2.imshow("Gray", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
