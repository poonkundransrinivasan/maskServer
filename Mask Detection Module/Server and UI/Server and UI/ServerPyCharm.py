from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import time
import cv2
import os
import pickle
import socket
import struct
import time
from socket import SOCK_SEQPACKET
from queue import Queue
import threading
import sqlite3 as db
import datetime
from multiprocessing import Process

from flask import Flask, render_template
import webbrowser
from flask_ngrok import run_with_ngrok

maskDB = db.connect("maskDB2.db")
cur = maskDB.cursor()
# cur.execute("CREATE TABLE IF NOT EXISTS maskData(srno INTEGER PRIMARY KEY AUTOINCREMENT, date DATE, time TIME, address TEXT, temp INTEGER, imageLoc TEXT)")
cur.execute(
    "CREATE TABLE IF NOT EXISTS maskData(srno INTEGER PRIMARY KEY AUTOINCREMENT, date DATE, time TIME, address TEXT, temp TEXT,reason TEXT, imageLoc TEXT)")
maskDB.commit()


def checkMask(frame):
    prototxtPath = r"C:\Users\vasu2\Desktop\BE Project\Mask_Detection_BE\face_detector\deploy.prototxt"
    weightsPath = r"C:\Users\vasu2\Desktop\BE Project\Mask_Detection_BE\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    maskNet = load_model(r"C:\Users\vasu2\Desktop\BE Project\Mask Detection Test\maskdata\MaskDetectionModel")

    frame = imutils.resize(frame, width=400)
    assert not isinstance(frame, type(None)), 'frame not found'
    preds = detect_and_predict_mask(frame, faceNet, maskNet)

    pred = zip(preds)

    try:
        str = list(pred)[0][0]
    except:
        return 3

    if str[0] > str[1]:
        return 1
    else:
        return 2


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return preds


PORT = 8090
global conn
hName = socket.gethostname()
HOST = socket.gethostbyname(hName)
a = "MASK!"
b = "NO MASK!"
c = "FACE NOT DETECTED!"
result = ""


def createSocket():
    global s
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')
    except socket.error as e:
        print("Error: " + str(e))


def socketBind():
    global s
    try:
        s.bind((HOST, PORT))
        print('Socket bind complete')
        s.listen(10)
        conn, address = s.accept()
        print("3")
        print(f"Connection from {address} has been established.")
    except socket.error as e:
        print("Error: " + str(e))


def socketListen():
    global s
    global conn
    global address

    s.listen(10)

    conn, address = s.accept()

    print(f"Connection from {address} has been established.")


def reteriveData():
    global s
    global conn
    global address
    global data
    global frame_data
    rawCode = ''
    code = ''
    rawTemp = ''
    temp = ''

    rawCode = conn.recv(8)
    code = rawCode.decode()

    data = b''  ### CHANGED
    payload_size = struct.calcsize("L")  ### CHANGED
    # Retrieve message size
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]  ### CHANGED

    # Retrieve all data based on message size
    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    print(code)
    if code == 'x':
        extractFrame()
        sendReply()

    else:
        # rawTemp = conn.recv(8)

        # temp = rawTemp.decode()
        conn.close()
        storeData(frame_data, code)


def storeData(frame_data, temp):
    global address
    frame = pickle.loads(frame_data)
    print(temp)
    path = r"C:\Users\vasu2\Desktop\BE Project\Mask Detection Test\maskdata\static"
    dateTimeList = []
    dateTimeList = str(datetime.datetime.now()).split(" ")
    loc = dateTimeList[0] + "-" + dateTimeList[1].split(".")[0].replace(":", "-") + ".bmp"
    print(cv2.imwrite(str(os.path.join(path, loc)), frame))
    reas = ""
    if (int(temp) > 99.9):
        reas = "High Temprature!"
    else:
        reas = "No Mask!"
    cur.execute("INSERT INTO maskData(date,time,address,temp,reason,imageLoc) VALUES (?,?,?,?,?,?);",
                (dateTimeList[0], dateTimeList[1], str(address), str(reas), str(temp), loc))
    maskDB.commit()
    showDataBase()


def showDataBase():
    cur.execute("SELECT * FROM maskData")
    while True:
        record = cur.fetchone()
        if record == None:
            break
        print(record)


def extractFrame():
    global s
    global frame_data
    global result
    # Extract frame
    t0 = time.time()
    frame = pickle.loads(frame_data)

    t1 = time.time()
    i = checkMask(frame)
    print("\nPickle Time: ", t1 - t0)
    print("\nCheck Time: ", time.time() - t1, "\n\n")

    print(i)
    if i == 1:
        result = a
    elif i == 2:
        result = b
    elif i == 3:
        result = c


def sendReply():
    global conn
    global address
    global result
    print(result)
    # conn.send(bytes(result,"utf-8"))
    conn.send(result.encode())
    conn.close()


def executeSystem():
    createSocket()
    socketBind()

    while True:
        socketListen()
        reteriveData()

    # reateThreads()
    # reateJobs()


#def frontend():
# global data
# #app = Flask(__name__)
# # run_with_ngrok(app)
#
# headings = ("DATE", "TIME", "ADDRESS", "TEMPERATURE", "REASON", "IMAGE")
#
#
# img = ".bmp"
#
# @app.route("/home")
# @app.route("/")
# def home():
#     cur.execute("SELECT date,time,address,reason,temp,imageLoc FROM maskData")
#     data = cur.fetchall()
#     return render_template("home.html", Headings=headings, Data=data, images=img)
#
# @app.route("/AboutUs")
# def AboutUs():
#     return render_template("AboutUs.html", Headings=headings, Data=data, images=img)
#
# @app.route("/ContactUs")
# def ContactUs():
#     return render_template("ContactUs.html", Headings=headings, Data=data, images=img)

# app.run(host="0.0.0.0")

def getSystemUp():

    p2 = Process(target=executeSystem)
    p2.start()
    # p1 = Process(target=app.run(host="0.0.0.0"))
    # p1.start()
    p2.join()
   # p1.join()



if __name__ == "__main__":

    print("Executing")
    executeSystem()