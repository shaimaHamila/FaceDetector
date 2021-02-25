import cv2
from random import randrange
# Load some pre-trained data on face from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
# img = cv2.imread('img11.jpg')
# img = cv2.imread('img7.png')

# To capture Video from webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    # 2 : for the thickness of the rectangle
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(255), randrange(255), randrange(255)), 2)

    # Display the image with the faces
    cv2.imshow('Chaima Face Detector',  frame)
    key = cv2.waitKey(1)  # We need waitKey to display something

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break


# Release the VideoCapture object
webcam.release()





