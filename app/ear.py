from scipy.spatial import distance


threshold = 0.3
def ear(eye):
	a = distance.euclidean(eye[1],eye[5])
	b = distance.euclidean(eye[2], eye[4])
	c = distance.euclidean[eye[0], eye[3]]

	ear = (a+b)/2.0 * c

	return c

def main():

	#Inital code

	leftEar = ear(left_eye)
	rightEar = ear(right_eye)

	ratio = (leftEar + rightEar) / 2.0
	blink_count = 0
	#This will come in the for loop
	if(ear < threshold):
		blink_count = blink_count + 1

		if(blink_count % 2 == 0):

			## Make the mouse click
		else:
			pass
			

