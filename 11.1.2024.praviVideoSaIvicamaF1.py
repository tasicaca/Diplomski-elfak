import numpy as np
import time
import cv2 as cv

# Dodaje ivice ivicnjaka i bele linije u izlaznom videu
# Object detection function (remains unchanged)
def pronadjiObjekte(outputs, image):
    hT, wT, cT = image.shape
    bounding_boxes = []
    class_ids = []
    confidences = []

    target_classes = ["car", "motorbike", "bus", "truck"]  # samo ove vrste yolo detektovanih objekata će se obrađivati.
    # Na početku sam imao dosta problema zbog detekcije objekata koji ne pripadju vozilima

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confThreshold and classNames[class_id] in target_classes:
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - w / 2), int((detection[1] * hT) - h / 2)
                bounding_boxes.append([x, y, x + w, y + h])  # Append the bounding box as a list
                class_ids.append(class_id)
                confidences.append(float(confidence))
                # https://stackoverflow.com/questions/58621583/how-find-confidence-for-each-classes-in-yolo-darknet
    bounding_boxes = np.array(bounding_boxes)

    indices = cv.dnn.NMSBoxes(bounding_boxes.tolist(), confidences, confThreshold, nmsThreshold)

    objects = []
    for i in indices:
        i = i[0] if isinstance(i, list) else i  # Izvlačenje podataka iz liste, ako postoje
        x, y, x2, y2 = bounding_boxes[i]
        w, h = x2 - x, y2 - y
        objects.append((x, y, w, h, False, True))
        cv.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)  # Zeleni pravougaonik preko objekta
        cv.putText(image, f'{classNames[class_ids[i]].upper()} {int(confidences[i] * 100)}%',
                   (x, y - 10), cv.FONT_ITALIC, 0.6, (0, 255, 0), 2)  # Zeleni tekst

    return objects


# Video capture and main processing loop
# https://www.irjmets.com/uploadedfiles/paper/issue_5_may_2022/23299/final/fin_irjmets1652765861.pdf
# https://github.com/mohamedameen93/Lane-lines-detection-using-Python-and-OpenCV/blob/master/Lane-Lines-Detection.ipynb
cap = cv.VideoCapture('granicestazeKatar.mp4') ###################madjarskagranicestaze.mp4######################################################ULAZNI VIDEO
x = 0  # inicijalizacija
y = 0
w = 0
h = 0

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'mp4v')

out = cv.VideoWriter('outputSaBELIMIVICNJACIMA.mp4', fourcc, 20.0, (width, height))
while True:
    current_time = time.time()
    # https://stackoverflow.com/questions/71664323/timer-update-inside-while-loop-if-statement
    success, frame = cap.read()
    if not success:
        break

    hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #######################################################dodato 1.4.2023.
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #  Gaussian Blur za smanjenje šuma
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Canny detektor ivica https://pyimagesearch.com/2021/05/12/opencv-edge-detection-cv2-canny/
    edges = cv.Canny(blurred, 50, 150)

    # Konvertuje sliku koja je crno bela u rgb vrednosti
    edges_3_channel = cv.merge([edges, edges, edges])

    # bitska operacija da se preko videa prikazu detektovane ivice (da bi sve bilo na jednom videu)
    posleBitskeOperacije_frame = cv.bitwise_or(frame, edges_3_channel)
    out.write(posleBitskeOperacije_frame)

    cv.imshow('Video with Edges', posleBitskeOperacije_frame)
    cv.imshow('Track limits, turn 4', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()



