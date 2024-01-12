import pyttsx3
from tkinter import *
import cv2
import ctypes
import numpy as np
import face_recognition  #face recognition library  it recognice and minupolate faces from python using
                        #deep learning model it has accuracy of 99.38%
import os
from datetime import datetime
from PIL import ImageTk  #python imaging library
import time

from PIL.ImageTk import PhotoImage

ctypes.windll.shcore.SetProcessDpiAwareness(1)  # it increase the window clearity
master = Tk()
master.geometry("1920x1080+-8+-8")

iconimg = PhotoImage(file="icon/cctv.png")
master.iconphoto(False, iconimg)  #icon set koribole
master.title('SMART SURVEILLANCE SYSTEM (Project by: Saklain)') #title for window
master.bind("<Escape>", exit)  # exit hb karone Esc click korile hb

bg = ImageTk.PhotoImage(file="icon/surviellance_bg2.png") #add background image
lblbg = Label(master, image = bg)
lblbg.place(x=0, y=0)

im = ImageTk.PhotoImage(file='icon/cu-logo.png')
leblimg=Label(master, image=im, width=250, height=300, bg='white')
leblimg.place(x=1600, y=650)

imglogo: PhotoImage = ImageTk.PhotoImage(file='icon/cctv2.png')
leblimg1=Label(master, image=imglogo, width=250, height=230, bg='white')
leblimg1.place(x=25, y=25)

#img2 = ImageTk.PhotoImage(file='icon/click-here1.png')
#leblimg2=Label(master, image=img2, width=100, height=90, bg='black')
#leblimg2.place(x=980, y=352)


path = 'images'
images = []
personNames = []
myList = os.listdir(path)
#print(myList)

for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
#print(personNames)

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = faceEncodings(images)


def storeit(name):
    with open('datastore.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')




#camera open and matching face from datastore
def CameraOn():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

        facesCurrentFrame = face_recognition.face_locations(faces)
        encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

        for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = personNames[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                storeit(name)

            else:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imwrite("unknown/unknown.png", frame)
                cap.release()
                popwin()

        cv2.imshow('FACE CAMERA', frame)
        if cv2.waitKey(1) == 13:
            break

    cap.release()


def windowexit():
    exit()


#########popup screen######################
def popwin():
    engine = pyttsx3.init()
    engine.say("who  are  you?")
    top = Toplevel(master)
    top.geometry("900x300")
    top['bg'] = 'yellow'
    engine.runAndWait()
    global lblr
    lblr = Label(top, text="YOUR FACE IS NOT REGISTERED", fg='red',  font=("times new roman", 25), bg='yellow')
    lblr.place(x=10, y=2)
    global lbl2
    lbl2 = Label(top, text="Please Enter Your Name", fg='black', font=("times new roman", 25), bg='yellow')
    lbl2.place(x=10, y=100)

    global name
    large_font = ('Verdana', 20)
    name = Entry(top, width=20, font=large_font, fg='DarkOrange3')  # this create entry box to write name
    name.grid(row=0, column=1)
    name.place(x=380, y=100)
    name.focus_set()

    def Run():

        global name_id

        if (name.get() == ""):
            print()

        else:
            name_id = name.get()

            name.delete(0, END)


        timer = int(5)  # timer
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow('NewFace Capture', frame)
            prev = time.time()

            while timer >= 0:
                ret, frame = cap.read()
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, str(timer), (200, 250), font, 7, (0, 255, 255), 4, cv2.LINE_AA)
                cv2.imshow('NewFace Captur', frame)
                # cv2.moveWindow('Video', 1200, 160)
                cv2.waitKey(1)
                cur = time.time()

                if cur - prev >= 1:
                    prev = cur
                    timer = timer - 1

            else:
                ret, frame = cap.read()
                cv2.imshow('NewFace Captur', frame)
                # cv2.moveWindow('Video', 1200, 160)
                cv2.waitKey(1)
                cv2.imwrite('images/' + name_id + '.jpg', frame)

            cap.release()

            cv2.destroyAllWindows()
            btn5.destroy()
            lbl2.destroy()
            lblr.destroy()
            name.destroy()
            top.destroy()


    global btn5
    btn5 = Button(top, text='capture !', bg='red', bd='10',command=Run)  # this button help to enroll the face of the student
    btn5.place(x=500, y=230)

#btn1 = Button(master, text='Register', bd='10', command=popwin)
#btn1.place(x=50, y=330)


###Lebel aru Button Home Pager###
label1 = Label(master, text = "YOU ARE UNDER SURVEILLANCE AREA", fg='green', font=("times new roman", 50), bg='yellow')
label1.place(x=300, y=80)

label2 = Label(master, text = "FOR SMART SECURITY START THE SYSTEM", fg='dark green', font=("times new roman", 20), bg='pink')
label2.place(x=650, y=300)

button = Button(master, text ='START SURVEILLANCE', bg='green', font=("times new roman", 20), command = CameraOn, bd = '15' )
button.place(x=800, y=460)

button1 = Button(master, text ='EXIT', fg='dark green', command = windowexit, bg='red', font=("times new roman", 20),  bd = '13')
button1.place(x=120, y=820)

cv2.destroyAllWindows()

mainloop()