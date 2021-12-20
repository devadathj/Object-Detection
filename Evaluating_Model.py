import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def IOU(boxA, boxB): #Function to calculate Intersection over Union
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    
    # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

net = cv2.dnn.readNetFromDarknet("yolov3_custom.cfg", r"yolov3_custom_last.weights")

classes = ['Dog', 'Cat', 'Elephant']

IOUthresh = 0.5

TP = [0]*3
FP = [0]*3
FN = [0]*3
Precision = [0]*3
Recall = [0]*3
F1 = [0]*3


impath = glob.glob('test_data\*jpg')
valpath = glob.glob('test_data\*txt')
for i in range(len(impath)):  
    valp = open(valpath[i]).read() #Reading and extracting the ground truth data
    valp = str.split(valp)
    valp = [valp[i:i + 5] for i in range(0, len(valp), 5)] 
    
    img = cv2.imread(impath[i])
    hight,width,_ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB = True, crop = False)
    
    net.setInput(blob)
    
    output_layers_name = net.getUnconnectedOutLayersNames()
    
    layerOutputs = net.forward(output_layers_name)
    
    boxes = []
    confidences = []
    class_ids = []
    
    for output in layerOutputs: #Assesing all the detections
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.5: #Acceptable confidence level for a detection
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3] * hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)
    # font = cv2.FONT_HERSHEY_PLAIN #For plotting the detections on the original image
    # colors = np.random.uniform(0,255,size =(len(boxes),3))
    
    if  len(indexes) > 0: #If there are any detections on the image
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            # color = colors[i] #For plotting the detections on the original image
            # cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            # cv2.putText(img,label + " " + confidence, (x,y+400),font,2,color,2)
        
        imgset = set(class_ids)
        valset = set([int(item[0]) for item in valp])
        if (imgset & valset): #Check if there are any correct detections
            imgset = set()
            valset = set()
            IOUmat = list()
            valboxes = list()
            
            for innerlist in boxes: #Converting the detections to required format
                innerlist[2] = innerlist[0] + innerlist[2]
                innerlist[3] = innerlist[1] + innerlist[3]
       
            for i in range(len(valp)): #Converting the ground truth to required format
                valw = round(float(valp[i][3])*width)
                valh = round(float(valp[i][4])*hight)
                valx = round((float(valp[i][1])*width) - valw/2)
                valy = round((float(valp[i][2])*hight) - valh/2)
                valboxes.append([valx, valy, valx + valw, valy + valh])
    
            for i in range(len(class_ids)):
                for j in range(len(valp)):
                    if class_ids[i] == int(valp[j][0]): #Calculating the IOU between all the detections and ground truths of same class 
                        IOUtemp = IOU(boxes[i], valboxes[j])
                        if IOUtemp >= IOUthresh: #Check if IOU is accepable
                            IOUmat.append([i, j, IOUtemp])
            
            IOUmat = sorted(IOUmat, key = lambda x: x[2], reverse = True)
            for i in range(len(IOUmat)):
                if IOUmat[i][0] not in imgset and IOUmat[i][1] not in valset: #Make sure only one detection per ground truth is counted.
                    imgset.add(IOUmat[i][0])
                    valset.add(IOUmat[i][1])
                    TP[class_ids[IOUmat[i][0]]] += 1
            
            for i in range(len(class_ids)): #For remaining detections
                if i not in imgset:
                    FP[class_ids[i]] += 1
                    
            for i in range(len(valp)): #For undetected ground truths
                if i not in valset:
                    FN[int(valp[i][0])] += 1
                
        else: #If there are no correct detections, every detection is a false positive and every ground truth is a false negative
            for i in range(len(class_ids)):
                FP[class_ids[i]] += 1
            for i in range(len(valp)):
                FN[int(valp[i][0])] += 1
            
    else: #If there are no detections on the input image, then every ground truth is a false negative
        for i in range(len(valp)):
            FN[int(valp[i][0])] += 1

    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
    
for i in range(0,3):
    Precision[i] = round(TP[i]/(TP[i] + FP[i]), 2)
    Recall[i] = round(TP[i]/(TP[i] + FN[i]), 2)  
    F1[i] = round(2*(Precision[i]*Recall[i])/(Precision[i] + Recall[i]), 2)

    
        