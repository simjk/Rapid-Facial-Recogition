import cv2
import os

def add_training_data():
	user=input("Enter the name of the person \n")
	while os.path.exists("image/train/"+user):
	    print('User with this name already exist. Please try again with new name or press ctrl+c to quit')
	    user=input("Enter the name of the person \n")
	os.makedirs("image/train/"+user)
	cap = cv2.VideoCapture(0)
	# Check success
	if not cap.isOpened():
	    raise Exception("Could not open video device")
	# Read picture. ret === True on success
	count=0
	while count<10:
	    ret, frame = cap.read()
	    cv2.imwrite("image/train/"+user+"/"+str(count)+".jpg",frame)
	    count=count+1
	    # Close device
	cap.release()

if __name__ == "__main__":
	add_training_data()