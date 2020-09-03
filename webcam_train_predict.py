import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import cv2
import numpy as np
import sys
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import PySimpleGUI as sg

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
	X = []
	y = []

	#Loop through each person in the training set
	for class_dir in os.listdir(train_dir):
		if not os.path.isdir(os.path.join(train_dir, class_dir)):
			continue

		#Loop through each training image for the current person
		for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
			image = face_recognition.load_image_file(img_path)
			face_bounding_boxes = face_recognition.face_locations(image)
			#print("face_bounding_boxes: ", face_bounding_boxes)


			if len(face_bounding_boxes) != 1:
				#if there are no people (or too many people) in a training image, skip the image
				if verbose:
					print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 
						else "Found more than one face"))
			else:
				#Add face encoding for current image to the training set
				X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
				y.append(class_dir)

	#Determine how many neighbors to use for weighting in the KNN classifier
	if n_neighbors is None:
		n_neighbors = int(round(math.sqrt(len(X))))
		if verbose:
			print("Chose n_neighbors automatically: ", n_neighbors)


	#Create and train the KNN classifier
	knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
	knn_clf.fit(X,y)

	#Save the trained KNN classifier
	if model_save_path is not None:
		with open(model_save_path, 'wb') as f:
			pickle.dump(knn_clf, f)

	return knn_clf


def predict(frame, knn_clf=None, model_path=None, distance_threshold=0.4):
	#distance_threshold: the lower the number, the stricter the check is
	if knn_clf is None and model_path is None:
		raise Exception("Must supply knn classifier either through knn_clf or model_path")

	#Load a trained KNN model
	if knn_clf is None:
		with open(model_path, 'rb') as f:
			knn_clf = pickle.load(f)

	X_face_locations = face_recognition.face_locations(frame)

	#If no faces are found in the image, return an empty result
	if len(X_face_locations) == 0:
		return []

	#Find encodings for faces in the test image
	faces_encodings = face_recognition.face_encodings(frame, known_face_locations=X_face_locations)

	#Use the KNN to find the best mathes for the test face
	closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)


	are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]


	#print("are_matches", [closest_distances[0][i][0] for i in range(len(X_face_locations))])

	#Predict classes and remove classification aren't within the threshold
	

	result = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

	#print ("result: ", result)
	return result


def show_prediction_labels_on_webcam(RGBFrame, predictions):	
	frame = RGBFrame
	name=""
	for name, (top, right, bottom, left) in predictions:
		#Scale back up face location
		# top *= 4
		# right *= 4
		# bottom *= 4
		# left *= 4

		# #Draw a rectangle over the face
		# cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

		# #Draw a label 
		# cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0,255,0),cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		# cv2.putText(frame, name, (left+ 6, bottom - 6), font, 1.0, (255,255,255), 1)
		
	#Display the resulting image
	#scv2.imshow("Face Recognition", frame)
	return name


def train_knn():
	print("Training KNN classifier...")
	classifier = train("image/train", model_save_path="trained_knn_model.clf", n_neighbors = 2)
	print("Training complete!")

# if __name__ == "__main__":
# 	#Train KNN classifier and save it to disk
# 	# print("Training KNN classifier...")
# 	# classifier = train("image/train", model_save_path="trained_knn_model.clf", n_neighbors = 2)
# 	# print("Training complete!")

# 	#Get a reference to webcam
# 	cap = cv2.VideoCapture(0)

# 	while True:
# 		ret, frame = cap.read()
# 		ret, frame1 = cap.read()

# 		#Resize to 1/4 size for faster face recognition processing
# 		small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

# 		#Convert the image from BGR to RGB
# 		rgb_small_frame = small_frame[:,:,::-1]
# 		#only process every other frame of vdieo to save time
		
# 		# #Use trained classifier, make prediction for unknown images
# 		#Find all the people in the image using a trained classifier model
# 		predictions = predict(rgb_small_frame, model_path="trained_knn_model.clf")

# 		#Display results overlaid on webcam video
# 		name = show_prediction_labels_on_webcam(frame, predictions)

# 		key = cv2.waitKey(1) & 0xFF
# 		if key == 27:
# 			break

# 		# if name == "unknown":
# 		# 	ans = input("Unknown detected. Do you want to add him/her?")
# 		# 	if ans == "y":			
# 		# 		add_training_data()
# 		# 		train_knn()
# 		# 	elif ans == "n":
# 		# 		continue



# 	#Release handle to webcam
# 	cap.release()
# 	cv2.destroyAllWindows()

