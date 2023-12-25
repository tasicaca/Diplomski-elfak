import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture('FerrariTrackLimits.mp4')
whT = 320
confThreshold = 0.5  # Adjusted confidence threshold for stricter detection
nmsThreshold = 0.3  # Adjusted NMS threshold for stricter detection

# LOAD YOLO MODEL
classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Define line coordinates and sensitivity (adjust as needed)
line_coordinates = [(100, 300), (500, 300)]
sensitivity = 1
line_thickness = 2

# Define the time threshold for showing the red light
red_light_threshold = 0  # in seconds

# Adjusted sensitivity for object leaving the road
leave_road_threshold = 0  # in seconds

def find_objects(outputs, image):
    hT, wT, cT = image.shape
    bounding_boxes = []
    class_ids = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confThreshold:
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - w / 2), int((detection[1] * hT) - h / 2)
                bounding_boxes.append([x, y, w, h])
                class_ids.append(class_id)
                confidences.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bounding_boxes, confidences, confThreshold, nmsThreshold)

    objects = []
    for i in indices:
        frame = bounding_boxes[i]
        x, y, w, h = frame[0], frame[1], frame[2], frame[3]
        objects.append((x, y, w, h, True))
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangles
        cv.putText(image, f'{classNames[class_ids[i]].upper()} {int(confidences[i] * 100)}%',
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Green text

    return objects

def pronadji_put(slika, objekti):
    # Convert the frame to grayscale
    siva_slika = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)

    # Use Canny edge detector
    edges = cv.Canny(siva_slika, 50, 150)

    # Create an empty mask
    maska_puta = np.zeros_like(siva_slika)

    # Set the region inside the detected edges as the road
    maska_puta[edges > 0] = 255

    konture, _ = cv.findContours(maska_puta, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    konture_puta = [kontura for kontura in konture if cv.contourArea(kontura) > 1000]

    # Identifikacija bele linije s jedne strane puta
    for kontura in konture_puta:
        aproksimacija = cv.approxPolyDP(kontura, 0.02 * cv.arcLength(kontura, True), True)
        if len(aproksimacija) == 2:
            cv.drawContours(slika, [kontura], -1, (255, 255, 255), 2)  # White color for the line
        else:
            cv.drawContours(slika, [kontura], -1, (128, 0, 128), 2)  # Purple color for road edges

    return konture_puta, maska_puta

prelazeci_objekti = []

while True:
    current_time = time.time()  # Get the current time

    success, frame = cap.read()
    if not success:
        break

    blob = cv.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_names = [(layer_names[i - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_names)

    objects = find_objects(outputs, frame)
    road_contours, road_mask = pronadji_put(frame, objects)

    # Update prelazeci_objekti based on whether an object intersects with the road contours
    for i, obj in enumerate(objects):
        x, y, w, h, _ = obj
        obj_center = (x + w // 2, y + h // 2)

        # Increase the strictness for leaving the road detection
        on_road = False
        for contour in road_contours:
            if cv.pointPolygonTest(contour, obj_center, False) >= 0:
                on_road = True
                break

        # Save the frame when an object leaves the road
        if not on_road:
            cv.imwrite(f'frame_{int(current_time)}.png', frame)

        objects[i] = obj[:4] + (on_road,)

    prelazeci_objekti = [i for i, obj in enumerate(objects) if not obj[-1]]

    for object_index in prelazeci_objekti:
        x, y, w, h, _ = objects[object_index]
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow rectangles

    # Display red bar at the top of the screen when objects are detected outside the road
    if prelazeci_objekti:
        frame[:50, :] = [0, 0, 255]  # Red color
    else:
        frame[:50, :] = [0, 0, 0]  # Set to black if no objects are detected

    # Detekcija ivica puta i obelezavanje puta bojom
    hsv_slika = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Definisanje opsega boja za detekciju sive boje puta
    donja_siva = np.array([10, 10, 60], dtype=np.uint8)
    gornja_siva = np.array([159, 85, 159], dtype=np.uint8)

    # Filtriranje slike da se dobije samo siva boja puta
    maska_sive = cv.inRange(hsv_slika, donja_siva, gornja_siva)

    # Neka boja za obelezavanje puta (npr. plava)
    boja_puta = (255, 0, 0)

    # Oznaci put na frejmu
    frame[maska_sive == 255] = boja_puta

    cv.imshow('Frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


##https://www.computervision.zone/topic/download-files-4/