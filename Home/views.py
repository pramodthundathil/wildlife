# Inside views.py
from django.shortcuts import redirect, render
import threading
from django.http import StreamingHttpResponse
import cv2
import numpy as np
import time
import os
import smtplib
import imghdr
from email.message import EmailMessage
from playsound import playsound
from django.views.decorators import gzip
from .models import Animal

from django.core.mail import EmailMessage
from django.template.loader import render_to_string



# Define global variables
stop_video_feed = False

# Function to alert
def alert():
    threading.Thread(target=playsound, args=('Home/alarm.wav',), daemon=True).start()

# Function to send email
def send_email():
    # Sender_Email = "cepcentre007@gmail.com"
    # Reciever_Email = "alvinjoseph2022@gmail.com"
    # Password = 'sjcapvmtthbkdhuh'   #ENTER GOOGLE APP PASSWORD HERE

    # newMessage = EmailMessage()   
    # newMessage['Subject'] = "Animal Detected" 
    # newMessage['From'] = Sender_Email  
    # newMessage['To'] = Reciever_Email  
    # newMessage.set_content('An animal has been detected') 

    # with open('images/' + label + '.png', 'rb') as f:
    #     image_data = f.read()
    #     image_type = imghdr.what(f.name)
    #     image_name = f.name[7:]

    # newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    # with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    #     smtp.login(Sender_Email, Password) 
    #     smtp.send_message(newMessage)
    mail_subject = 'animal detected'
    message = render_to_string('emailbody.html', {'name': "Animal Detacted",})

    email = EmailMessage(mail_subject, message, to=["alvinjoseph2022@gmail.com","gopinath.pramod@gmail.com"])
    email.send(fail_silently=True)

# Function to asynchronously send email
def async_email(label):
    threading.Thread(target=send_email, args=(label,), daemon=True).start()

# Function to process frame
def process_frame():
    args = {"confidence":0.5, "threshold":0.3}
    labelsPath = "Home\yolo-coco\coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    final_classes = ['bird', 'cat', 'dog', 'sheep', 'horse', 'cow', 'elephant', 'Tiger', 'bear', 'giraffe']
    domestic_animals = ['bird', 'cat', 'dog', 'sheep', 'cow']
    wild_animals = ['elephant', 'Tiger', 'bear', 'giraffe']
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    weightsPath = os.path.abspath("Home\yolo-coco\yolov3-tiny.weights")
    configPath = os.path.abspath("Home\yolo-coco\yolov3-tiny.cfg")

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    vs = cv2.VideoCapture(0)
    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                if LABELS[classIDs[i]] in domestic_animals:
                    color = (0, 255, 0)  
                elif LABELS[classIDs[i]] in wild_animals:
                    color = (0, 0, 255)  
                else:
                    try:
                        val = Animal.objects.get(id = 1)
                        val.animal = LABELS[classIDs[i]]
                        val.save()
                    except:
                        Animal.objects.create(animal = LABELS[classIDs[i]]).save()
                    color = (255, 0, 0)  
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                try:
                    val = Animal.objects.get(id = 1)
                    val.animal = LABELS[classIDs[i]]
                    val.save()
                except:
                    Animal.objects.create(animal = LABELS[classIDs[i]]).save()

                if LABELS[classIDs[i]] in domestic_animals:
                    color = (0, 255, 0)  
                elif LABELS[classIDs[i]] in wild_animals:
                    mail_subject = 'animal detected'
                    message = render_to_string('emailbody.html', {'name': "Animal Detacted",})

                    email = EmailMessage(mail_subject, message, to=["alvinjoseph2022@gmail.com","gopinath.pramod@gmail.com"])
                    email.send(fail_silently=True)
                    print("---------------------------------------------------")
                    alert() 

                            # Alert if any animal detected
                    # async_email(LABELS[classIDs[i]])  # Send email
        else:
            flag=True  # This line seems unnecessary

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    vs.release()

def generate_frames():
    global stop_video_feed
    while not stop_video_feed:
        for frame in process_frame():
            yield frame

def Index(request):
    try:
        animal = Animal.objects.all()[0].animal
    except:
        animal = "No Animal"

    context = {
        "animal":animal
    }

    return render(request, "index.html",context)

def CallCam(request):
    global stop_video_feed
    stop_video_feed = False
    threading.Thread(target=generate_frames).start()
    return redirect('VideoMin')


# views.py

from django.http import JsonResponse
 

def get_animal_data(request):
    # Assuming you have a model named Animal with a field called 'name'
    # You can modify this query based on your actual model structure
    latest_animal = Animal.objects.latest('id')
    animal_name = latest_animal.animal
    return JsonResponse({'animal': animal_name})


def VideoMin(request):
    try:
        animal = Animal.objects.all()[0].animal
    except:
        animal = "No Animal"

    context = {
        "animal":animal
    }

    return render(request, "video_frame.html",context)

@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
