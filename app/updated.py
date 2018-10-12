from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("##TODO")

(left_start, left_end) = face_utils.FACIAL_LANDMARKS_68_IDXS(["left_eye"])
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_68_IDXS(["right_eye"])

cap = cv2.VideoCapture(0)

while True:

	frame, ret = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape2 = face_utils.shape_to_np(shape)
		left_eye = shape[left_start:left_end]
		right_eye = shape[right_star:right_end]
		left_eye_hull = cv2.convexHull(left_eye)
		right_eye_hull = cv2.convexHull(right_eye)
		cv2.drawContours(frame,[left_eye_hull], -1, (255,0,255), -1)
		cv2.drawContours(frame, [right_eye_hull], -1, (255,0,255), -1)
		
	cv2.imshow(frame,"Aankh")

cap.destroyAllWindows()
cap.stop()
