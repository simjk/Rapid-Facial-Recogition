import cv2, PySimpleGUI as sg
import os
import numpy as np
from webcam_train_predict import *

#train_knn()

layout = [[sg.Image(filename='')],
            [sg.Text('', justification='center', font='Helvetice 20', size=(10,1), key='output')],
            [sg.RButton('Add', size=(10,1), pad=((200, 0), 3), font='Any 14')]
            ]
#create the window and show it without the plot
window = sg.Window('Face Recognition', [[sg.Image(filename='', key='image')],], location=(800,400))

window.Layout(layout).Finalize()


cap = cv2.VideoCapture(0)       # Setup the camera as a capture device


while True:                     # The PSG "Event Loop"
    event, values = window.Read(timeout=20)      # get events for the window with 20ms max wait

    imgbytes = cv2.imencode('.png', cap.read()[1])[1].tobytes()
    window['image'].update(data=imgbytes)

    ret, frame = cap.read()

    #Resize to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)  
    #Convert the image from BGR to RGB
    rgb_small_frame = small_frame[:,:,::-1]
    #only process every other frame of video to save time  
    # #Use trained classifier, make prediction for unknown images
    #Find all the people in the image using a trained classifier model
    predictions = predict(rgb_small_frame, model_path="trained_knn_model.clf")

    name = show_prediction_labels_on_webcam(frame, predictions)

    if event is None:
        break                                            
        # if user closed window, quit
    elif event is 'Add':
        user = sg.PopupGetText('Enter the name of the person', 'Add Training Data')

        if os.path.exists("image/train/"+user):
            sg.Popup('User with this '+user+' already exist.', 'Please try again with new name or press ctrl+c to quit')
            #print('User with this name already exist. Please try again with new name or press ctrl+c to quit')
            user = sg.PopupGetText('Enter the name of the person', 'Add Training Data')
        os.makedirs("image/train/"+user)
        count=0
        while count<10:
            ret, frame = cap.read()
            cv2.imwrite("image/train/"+user+"/"+str(count)+".jpg",frame)
            count=count+1
        train_knn()
    elif event is not None:
        if name is not 'unknown':
            window['output'].update(name)
        else:
            name = 'unknown'
            window['output'].update(name)


         # if len(predictions) > 1:
        #     name_list = []
        #     for name, (top, right, bottom, left) in predictions:
        #         font = cv2.FONT_HERSHEY_DUPLEX
        #         name_list.append(name)
            
        #     print(len(predictions), ": ", name_list)
        #     window['output'].update(name_list)
        # else:

