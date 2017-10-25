# Cut and paste from:
#   http://dlib.net/face_recognition.py.html
#   https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py
#   https://medium.com/towards-data-science/facial-recognition-using-deep-learning-a74e9059a150
#
# Install dlib: See https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/

import os
import dlib
import numpy as np
from skimage import io
import cv2
from datetime import datetime

data_dir = os.path.expanduser('~/data')
faces_folder_path = data_dir + '/kodemaker/'

# Globals
face_detector_dlib = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(data_dir + '/dlib/shape_predictor_5_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1(data_dir + '/dlib/dlib_face_recognition_resnet_model_v1.dat')
face_classifier_opencv = cv2.CascadeClassifier(data_dir + '/opencv/haarcascade_frontalface_default.xml')


def timefn(fn, args):
    start = datetime.now()
    r = fn(*args)
    elapsed = datetime.now() - start
    print(fn.__name__ + " took ", elapsed)
    return r


def to_dlib_rect(w, h):
    return dlib.rectangle(left=0, top=0, right=w, bottom=h)

def to_rect(dr):
    #  (x, y, w, h)
    return dr.left(), dr.top(), dr.right()-dr.left(), dr.bottom()-dr.top()

def face_detector_opencv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_classifier_opencv.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE)


def get_face_encodings(face, bounds):
    # start = datetime.now()
    # bounds = face_detector(face, 1)
    # print("Dlib Getting bounds took ", (datetime.now() - start))
    start = datetime.now()
    faces_landmarks = [shape_predictor(face, face_bounds) for face_bounds in bounds]
    print("Getting shape predictors took ", (datetime.now() - start))
    return [np.array(face_recognition_model.compute_face_descriptor(face, face_pose, 1)) for face_pose in faces_landmarks]


def get_face_matches(known_faces, face):
    return np.linalg.norm(known_faces - face, axis=1)


def find_match(known_faces, person_names, face):
    matches = get_face_matches(known_faces, face) # get a list of True/False
    min_index = matches.argmin()
    min_value = matches[min_index]
    if min_value < 0.58:
        return person_names[min_index]+" ({0:.2f})".format(min_value)
    if min_value < 0.65:
        return person_names[min_index]+"?"+" ({0:.2f})".format(min_value)
    return 'Not Found'


def load_face_encodings(faces_folder_path):
    image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir(faces_folder_path))
    image_filenames = sorted(image_filenames)
    person_names = [x[:-4] for x in image_filenames]

    full_paths_to_images = [faces_folder_path + x for x in image_filenames]
    face_encodings = []

    win = dlib.image_window()

    for path_to_image in full_paths_to_images:
        face = io.imread(path_to_image)

        faces_bounds = face_detector_dlib(face, 0)

        if len(faces_bounds) != 1:
            print("Expected one and only one face per image: " + path_to_image + " - it has " + str(len(faces_bounds)))
            exit()

        face_bounds = faces_bounds[0]
        face_landmarks = shape_predictor(face, face_bounds)
        face_encoding = np.array(
            face_recognition_model.compute_face_descriptor(face, face_landmarks, 1)
        )


        win.clear_overlay()
        win.set_image(face)
        win.add_overlay(face_bounds)
        win.add_overlay(face_landmarks)
        face_encodings.append(face_encoding)

        print(face_encoding)

        # dlib.hit_enter_to_continue()
    return face_encodings, person_names


def recognize_faces_in_video(face_encodings, person_names):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = datetime.now()
        face_rects = face_detector_opencv(frame)
        print("CV2 Getting bounds took ", (datetime.now() - start))
        #start = datetime.now()
        #bounds = face_detector(frame, 1)
        #print("dlib Getting bounds took ", (datetime.now() - start))

        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = frame[y:y + h, x:x + w]

            bounds = ([to_dlib_rect(w.item(), h.item())])
            face_encodings_in_image = get_face_encodings(face, bounds)
            if (face_encodings_in_image):
                match = find_match(face_encodings, person_names, face_encodings_in_image[0])
                cv2.putText(frame, match, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Debugging dlib localizer:
            bounds = face_detector_dlib(frame, 0)
            if (bounds):
                b = bounds[0]
                cv2.rectangle(frame, (b.left(), b.top()), (b.right(), b.bottom()), (255, 255, 0), 2)
            face_encodings_in_image = get_face_encodings(frame, bounds)
            if (face_encodings_in_image):
                match = find_match(face_encodings, person_names, face_encodings_in_image[0])
                cv2.putText(frame, match, (b.left()+5, b.top()-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("bilde", frame)

        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


face_encodings, person_names = load_face_encodings(faces_folder_path)
recognize_faces_in_video(face_encodings, person_names)



