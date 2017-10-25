# Cut and paste from:
#   http://dlib.net/face_recognition.py.html
#   https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py
#   https://medium.com/towards-data-science/facial-recognition-using-deep-learning-a74e9059a150

import os
import dlib
import numpy as np
from skimage import io
import cv2

data_dir = os.path.expanduser('~/data')
predictor_path = data_dir + '/dlib/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = data_dir + '/dlib/dlib_face_recognition_resnet_model_v1.dat'
faces_folder_path = data_dir + '/kodemaker/'
opencv_classifier_folder_path = data_dir + '/opencv/'

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_recognition_model = dlib.face_recognition_model_v1(face_rec_model_path)

win = dlib.image_window()

# This is the tolerance for face comparisons
# The lower the number - the stricter the comparison
# To avoid false matches, use lower value
# To avoid false negatives (i.e. faces of the same person doesn't match), use higher value
# 0.5-0.6 works well
TOLERANCE = 0.6

# This function will take an image and return its face encodings using the neural network
def get_face_encodings(image):
    # Detect faces using the face detector
    detected_faces = face_detector(image, 1)
    # Get pose/landmarks of those faces
    # Will be used as an input to the function that computes face encodings
    # This allows the neural network to be able to produce similar numbers for faces of the same people, regardless of camera angle and/or face positioning in the image
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    # For every face detected, compute the face encodings
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]


# This function takes a list of known faces
def compare_face_encodings(known_faces, face):
    # Finds the difference between each known face and the given face (that we are comparing)
    # Calculate norm for the differences with each known face
    # Return an array with True/Face values based on whether or not a known face matched with the given face
    # A match occurs when the (norm) difference between a known face and the given face is less than or equal to the TOLERANCE value
    return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE)


# This function returns the name of the person whose image matches with the given face (or 'Not Found')
# known_faces is a list of face encodings
# names is a list of the names of people (in the same order as the face encodings - to match the name with an encoding)
# face is the face we are looking for
def find_match(known_faces, names, face):
    # Call compare_face_encodings to get a list of True/False values indicating whether or not there's a match
    matches = compare_face_encodings(known_faces, face)
    # Return the name of the first match
    count = 0
    for match in matches:
        if match:
            return names[count]
        count += 1
    # Return not found if no match found
    return 'Not Found'

# Get path to all the known images
# Filtering on .jpg extension - so this will only work with JPEG images ending with .jpg
image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir(faces_folder_path))
# Sort in alphabetical order
image_filenames = sorted(image_filenames)
# Get full paths to images
paths_to_images = [faces_folder_path + x for x in image_filenames]
# List of face encodings we have
face_encodings = []
# Loop over images to get the encoding one by one
for path_to_image in paths_to_images:
    # Load image using scipy
    image = io.imread(path_to_image)
    # Get face encodings from the image
    face_encodings_in_image = get_face_encodings(image)
    # Make sure there's exactly one face in the image
    if len(face_encodings_in_image) != 1:
        print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
        exit()
    # Append the face encoding found in that image to the list of face encodings we have
    face_encodings.append(face_encodings_in_image[0])

names = [x[:-4] for x in image_filenames]




faceClassifier = cv2.CascadeClassifier(opencv_classifier_folder_path + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)   # cv2.VideoCapture("./out.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = faceClassifier.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 50),
        flags = cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faceRects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = frame[y:y + h, x:x + w]
        face_encodings_in_image = get_face_encodings(face)
        if (face_encodings_in_image):
            match = find_match(face_encodings, names, face_encodings_in_image[0])
            cv2.putText(frame, match, (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("bilde", frame)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()



# ########################################
# ## TODO: Merge with similar code above, to get a nice visual on descriptors
#
# # Now, collect face descriptors
# for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
#     print("Processing file: {}".format(f))
#     img = io.imread(f)
#
#     win.clear_overlay()
#     win.set_image(img)
#
#     # Ask the detector to find the bounding boxes of each face. The 1 in the
#     # second argument indicates that we should upsample the image 1 time. This
#     # will make everything bigger and allow us to detect more faces.
#     dets = face_detector(img, 1)
#     print("Number of faces detected: {}".format(len(dets)))
#
#     # Now process each face we found.
#     for k, d in enumerate(dets):
#         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#             k, d.left(), d.top(), d.right(), d.bottom()))
#         # Get the landmarks/parts for the face in box d.
#         shape = shape_predictor(img, d)
#         # Draw the face landmarks on the screen so we can see what face is currently being processed.
#         win.clear_overlay()
#         win.add_overlay(d)
#         win.add_overlay(shape)
#
#         # Compute the 128D vector that describes the face in img identified by
#         # shape.  In general, if two face descriptor vectors have a Euclidean
#         # distance between them less than 0.6 then they are from the same
#         # person, otherwise they are from different people. Here we just print
#         # the vector to the screen.
#         face_descriptor = face_recognition_model.compute_face_descriptor(img, shape)
#         print(face_descriptor)
#         # It should also be noted that you can also call this function like this:
#         #  face_descriptor = facerec.compute_face_descriptor(img, shape, 100)
#         # The version of the call without the 100 gets 99.13% accuracy on LFW
#         # while the version with 100 gets 99.38%.  However, the 100 makes the
#         # call 100x slower to execute, so choose whatever version you like.  To
#         # explain a little, the 3rd argument tells the code how many times to
#         # jitter/resample the image.  When you set it to 100 it executes the
#         # face descriptor extraction 100 times on slightly modified versions of
#         # the face and returns the average result.  You could also pick a more
#         # middle value, such as 10, which is only 10x slower but still gets an
#         # LFW accuracy of 99.3%.
#
#
#         dlib.hit_enter_to_continue()