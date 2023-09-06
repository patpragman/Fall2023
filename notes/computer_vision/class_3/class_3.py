import cv2
from numpy import ndarray
from math import sqrt
import numpy as np

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

    # linear operators

    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])

    kernel1 = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    kernel2 = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]]).T



    while True:

        ret, frame = camera.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Gaussian blur for denoising
        kernel_size = (5, 5)
        frame = cv2.GaussianBlur(frame, kernel_size, 1)

        frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel1)
        frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel2)
        # frame = cv2.filter2D(src=frame, ddepth=-1, kernel=laplacian)

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
