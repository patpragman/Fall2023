import cv2
from numpy import ndarray
from math import sqrt


def draw_smile(frame_you_want_to_modify: ndarray,  # this is the opencv image piped in from the camera
               # but really, it's a numpy n-dimensional array.  so a tensor of "channels" just like
               # the examples in class that we had - it's got 3 channels, and it's in BGR format
               ):
    """
    this code is adapted from some of my previous work and from some geeks-4-geeks code I stole and have rearranged.
    It takes a frame, then draws some smiley faces on it over the people's faces it detects in the frames.
    """

    gray = cv2.cvtColor(frame_you_want_to_modify, cv2.COLOR_BGR2GRAY)  # convert the image to grayscale - note cv2.COLOR_BGR2GRAY

    # this returns a list of faces as coordinates
    faces = face_cascade.detectMultiScale(gray, # the gray scale image
                                          scaleFactor=1.35,
                                          minNeighbors=5)
    # the params in that function seem mysterious (and they are) there's a great explanation here:
    # https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php

    for (x, y, w, h) in faces:
        """
        now let's iterate through the faces detected with the cascade classifier
        
        the face detector spits out the faces as a tuple containing the coordinates and the width and height

        """

        center_x = x + w // 2
        center_y = y + h // 2
        r = int(sqrt((w // 2) ** 2 + (h // 2) ** 2) // 1)  # radius of the face as an integer

        # a couple of values to figure out where to stick the eyes
        k = .60
        j = k / 2

        eye_width = int((k * r) // 1)
        eye_height = int((j * r) // 1)

        # draw the head
        cv2.circle(frame_you_want_to_modify,
                   (center_x, center_y), r, (255, 0, 0), -1)

        # draw the mouth
        cv2.ellipse(frame_you_want_to_modify, (center_x, center_y),
                    (r - 10, r - 10), 0, 0, 180, (0, 0, 0), -1)

        # draw some eyes
        cv2.circle(frame_you_want_to_modify, (center_x - eye_width, center_y - eye_height), 20, (0, 0, 0), -1)
        cv2.circle(frame_you_want_to_modify, (center_x - eye_width, center_y - eye_height), 5, (255, 255, 255), -1)

        # draw the pupils of the eyes
        cv2.circle(frame_you_want_to_modify, (center_x + eye_width, center_y - eye_height), 20, (0, 0, 0), -1)
        cv2.circle(frame_you_want_to_modify, (center_x + eye_width, center_y - eye_height), 5, (255, 255, 255), -1)

    return frame_you_want_to_modify


if __name__ == "__main__":
    # load up a camera - you'll need to have a camera plugged in
    camera = cv2.VideoCapture(0)  # if you have multiple cameras, you can cycle through them here

    """if you get an error looking for the haarcascade classifier, you can download a one here but it should come with
    opencv, a nice blogpost on how it works is available in the same link as described in the function above about
    the classifier parameters:
    https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
        
    
    if something doesn't work you can download it here:  https://github.com/opencv/opencv/tree/master/data/haarcascades
    then point the next line of code at the cascade classifier - it should 
    """

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # if this doesn't work, try just using this as the argument:
    # cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        """
        this is the loop where we'll handle the camera input!  we'll just break out of it with q or esc key when you're
        ready to quit!
        """

        ret, frame = camera.read()
        """
        camera.read() returns a tuple, ret is whether a boolean saying if any information was returned from the camera
        and frame is the numpy ndarray that contains the information from the camera
        """

        frame = draw_smile(frame)  # draw smiley faces on our frame
        cv2.imshow('Happy People!', frame)  # go figure, "imshow" shows an image

        # open CV can handle some key press stuff too
        key_pressed = cv2.waitKey(1)  # this gives the current key, I don't know why it's set up this way
        if key_pressed == 27 or key_pressed == ord("q"):
            """
            there's a bunch of key codes you can look up for key press codes,
            in this case when we get a q or the esc key, break out of the loop
            27 is the escape key - I found it in a geeks-4-geeks post, but there
            is actually good documentation in the language of your choice in opencv's documentations
            """
            break

    camera.release()  # release the camera, it should do this anyway when the program ends, but yah

    cv2.destroyAllWindows()  # close and kill the windows now that you're done
