import cv2
import numpy as np

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def draw_smile(gray, frame):
    """
    this nice code comes from geeks4geeks stackoverflow and a mix of crazy tinkering
    """

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # iterate through the faces and draw a smiley face over every detected face
    for (x, y, w, h) in faces:
        # detect faces and draw a smiley

        center_x = x + w//2
        center_y = y + h//2
        r = int(np.sqrt((w//2)**2 + (h//2)**2)//1)
        k = .60
        j = k/2

        eye_width = int((k*r)//1)
        eye_height = int((j*r)//1)

        cv2.circle(frame,
                   (center_x, center_y), r, (255, 0, 0), -1)

        cv2.ellipse(frame, (center_x, center_y),
                    (r - 10, r - 10), 0, 0, 180, (0, 0, 0), -1)

        cv2.circle(frame, (center_x - eye_width, center_y - eye_height), 20, (0, 0, 0), -1)
        cv2.circle(frame, (center_x - eye_width, center_y - eye_height), 5, (255, 255, 255), -1)

        cv2.circle(frame, (center_x + eye_width, center_y - eye_height), 20, (0, 0, 0), -1)
        cv2.circle(frame, (center_x + eye_width, center_y - eye_height), 5, (255, 255, 255), -1)



    return frame




if __name__ == "__main__":

    while True:

        ret, frame = camera.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = draw_smile(img, frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

