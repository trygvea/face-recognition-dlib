# Cut and paste from:
#   http://dlib.net/face_recognition.py.html
#   https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py
#   https://medium.com/towards-data-science/facial-recognition-using-deep-learning-a74e9059a150

import os
import dlib
import numpy as np
from skimage import io
import cv2
from collections import namedtuple

data_dir = os.path.expanduser('~/data')
faces_folder_path = data_dir + '/kodemaker/'

# Globals
dlib_frontal_face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(data_dir + '/dlib/shape_predictor_5_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1(data_dir + '/dlib/dlib_face_recognition_resnet_model_v1.dat')
face_classifier_opencv = cv2.CascadeClassifier(data_dir + '/opencv/haarcascade_frontalface_default.xml')


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


def face_detector_dlib(image):
    bounds = dlib_frontal_face_detector(image, 0) # second parameter is upsample; 1 or 2 will detect smaller faces. 0 performs similar to opencv with current parameters
    return list(map(lambda b: to_rect(b), bounds))


def get_face_encodings(face, bounds):
    faces_landmarks = [shape_predictor(face, face_bounds) for face_bounds in bounds]
    return [np.array(face_recognition_model.compute_face_descriptor(face, face_pose, 1)) for face_pose in faces_landmarks]


def get_face_matches(known_faces, face):
    return np.linalg.norm(known_faces - face, axis=1)


def find_match(known_faces, person_names, face):
    matches = get_face_matches(known_faces, face) # get a list of True/False
    min_index = matches.argmin()
    min_value = matches[min_index]
    if min_value < 0.62:
        return (person_names[min_index], True, min_value)
    return ('Not Found', False, None)


def format_name(name, found, accuracy):
    if not found:
        return 'Not found'
    if accuracy < 0.58:
        return name + " ({0:.2f})".format(accuracy)
    return name + "? ({0:.2f})".format(accuracy)


def load_face_encodings(faces_folder_path):
    image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir(faces_folder_path))
    image_filenames = sorted(image_filenames)
    person_names = [x[:-4] for x in image_filenames]

    full_paths_to_images = [faces_folder_path + x for x in image_filenames]
    known_faces = []

    win = dlib.image_window()

    for path_to_image in full_paths_to_images:
        face = io.imread(path_to_image)

        faces_bounds = dlib_frontal_face_detector(face, 0)

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
        known_faces.append(face_encoding)

        # print(face_encoding)

        # dlib.hit_enter_to_continue()
    return known_faces, person_names


RecognizedFace = namedtuple('RecognizedFace', 'known rect name accuracy')


def recognize_faces(image, use_dlib_for_detection=False):
    """
    Recognise all faces in image, and return a list of (known, rect, name, accuracy) for each.
    If more than one face in image, these are sorted after size, biggest first.
    A face is not known if it is recognised as a face, but no matching face embeddings have been recorded.
    Rect is position within image, and accuracy is the accuracy (for known faces) returned from dlib
    faceNet, where about 0.6 is the boundary value between known and not known faces.
    """
    faces = list()
    face_rects = face_detector_opencv(image) if use_dlib_for_detection else face_detector_dlib(image)
    for (x, y, w, h) in face_rects:
        face_image = image[y:y + h, x:x + w]
        bounds = ([to_dlib_rect(w.item(), h.item())]) if use_dlib_for_detection else ([to_dlib_rect(w, h)]) # int32 when opencv
        face_encodings_in_image = get_face_encodings(face_image, bounds)
        if (face_encodings_in_image):
            name, known, accuracy = find_match(known_faces, person_names, face_encodings_in_image[0])
            faces.append(RecognizedFace(
                known=known,
                rect=(x, y, w, h),
                name=name,
                accuracy=accuracy
            ))
    return sorted(faces, key=lambda face: face.rect[2] * face.rect[3])


def recognize_faces_in_video(known_faces, person_names):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = recognize_faces(frame)

        for face in faces:
            (x, y, w, h) = face.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            formattedName = format_name(face.name, face.known, face.accuracy)
            cv2.putText(frame, formattedName, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("bilde", frame)

        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


known_faces, person_names = load_face_encodings(faces_folder_path)
recognize_faces_in_video(known_faces, person_names)



