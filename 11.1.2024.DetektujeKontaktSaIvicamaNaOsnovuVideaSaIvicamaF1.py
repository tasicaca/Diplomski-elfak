import numpy as np
import time
import cv2 as cv

# Konstante
whT = 320
confThreshold = 0.1
nmsThreshold = 0.1

# Load YOLO model
classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
modelConfig = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def dodirujeBelu(hsv_pixel):
    # hsv vrednosti okolnih piksela se analiziraju kako bi se utvrdilo
    # da li je yolo objekat došao u zonu potpuno belih piksela kojima su obelezni ivicnjaci
    radius = 2   #6 daje dva za madjarsku
    tolerance = 10 #10 daje dva za madjarsku
    h_center, s_center, v_center = hsv_pixel
    is_almost_white = False

    # Provera da li su pikseli u neposrednoj okolini beli
    for h in range(h_center - radius, h_center + radius + 1):
        for s in range(s_center - radius, s_center + radius + 1):
            for v in range(v_center - radius, v_center + radius + 1):
                # Uvedena je tolerancija zbog nepreciznosti detekcije yolo objekta
                is_almost_white = (abs(h - 0) <= tolerance or abs(s - 0) <= tolerance or abs(v - 255) <= tolerance)

                if is_almost_white:
                    return True  # Pronadjen piksel, vraca se true

    return False  # Nema potencijalnih kandidata u blizini

def touching_whitebezR(hsv_pixel):

    blue_hsv_range = [(0, 0), (0, 0), (255, 255)]

    is_white = all(H_MIN <= hsv_pixel[i] <= H_MAX for i, (H_MIN, H_MAX) in enumerate(blue_hsv_range))

    return is_white

# Detekcija objekta
def dodirujePlavu(hsv_pixel):
    """blue_hsv_range = [(240, 240), (255, 255), (255, 255)]
    # Provera da li je piksel potpuno plav
    is_blue = all(H_MIN <= hsv_pixel[i] <= H_MAX for i, (H_MIN, H_MAX) in enumerate(blue_hsv_range))
    return is_blue

    """
    radius = 5 #5 daje dva za madjarsku
    tolerance = 20 # 10 daje 2 za madj
    h_center, s_center, v_center = hsv_pixel
    is_almost_blue = False

    # Provera da li su pikseli u neposrednoj okolini plavi
    for h in range(h_center - radius, h_center + radius + 1):
        for s in range(s_center - radius, s_center + radius + 1):
            for v in range(v_center - radius, v_center + radius + 1):
                # Uvedena je tolerancija zbog nepreciznosti detekcije yolo objekta
                is_almost_blue = (abs(h - 240) <= tolerance or abs(s - 255) <= tolerance or abs(v - 255) <= tolerance)

                if is_almost_blue:
                    return True  # Pronadjen piksel, vraca se true

    return False  # Nema potencijalnih kandidata u blizini


def pronadjiObjekte(outputs, image):
    hT, wT, cT = image.shape
    bounding_boxes = []
    class_ids = []
    confidences = []

    target_classes = ["car", "motorbike", "bus", "truck"]  # samo ove vrste yolo detektovanih objekata će se obrađivati.
    # Na početku sam imao dosta problema zbog detekcije objekata koji ne pripadaju vozilima

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
                   (x, y - 5), cv.FONT_ITALIC, 0.6, (0, 255, 0), 2)  # Zeleni tekst

    return objects

# https://github.com/mohamedameen93/Lane-lines-detection-using-Python-and-OpenCV/blob/master/Lane-Lines-Detection.ipynb
# https://www.irjmets.com/uploadedfiles/paper/issue_5_may_2022/23299/final/fin_irjmets1652765861.pdf
cap = cv.VideoCapture('outputSaBELIMIVICNJACIMAiPLAVIMPUTEM.mp4')
#fourcc = cv.VideoWriter_fourcc(*'XVID')
#out = cv.VideoWriter('izlazakSaStaze.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    current_time = time.time()
    # https://stackoverflow.com/questions/71664323/timer-update-inside-while-loop-if-statement
    success, frame = cap.read()
    if not success:
        break

    hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    blob = cv.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)  # https://programtalk.com/python-more-examples/cv2.cv2.dnn.blobFromImage/
    layer_names = net.getLayerNames()
    output_names = [(layer_names[i - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_names)  # https://www.grepper.com/answers/526529/output_layers+%3D+%5Blayer_names%5Bi%5B0%5D+-+1%5D+for+i+in+net.getUnconnectedOutLayers%28%29%5D+IndexError%3A+invalid+index+to+scalar+variable.
    # https: // www.computervision.zone / projects /

    objekti = pronadjiObjekte(outputs, frame)

    on_blue = False
    on_white = False
    for i, obj in enumerate(objekti):
        x, y, w, h, _, on_road = obj[:6]
        bottom_center_x = x + w // 2
        bottom_center_y = y + h
        bottom_center = (bottom_center_x, bottom_center_y)

        if 0 <= bottom_center_x < hsv_image.shape[0] and 0 <= bottom_center_y < hsv_image.shape[1]:
            hsv_bottom_center = hsv_image[bottom_center_x, bottom_center_y]
        else:
            print("Nepravilni opseg indeksa!")

        bottom_edge_left_x = x
        bottom_edge_left_y = y + h
        bottom_edge_left = (bottom_edge_left_x, bottom_edge_left_y)

        if 0 <= bottom_edge_left_x < hsv_image.shape[0] and 0 <= bottom_edge_left_y < hsv_image.shape[1]:
            hsv_bottom_edge_left = hsv_image[bottom_edge_left_x, bottom_edge_left_y]
        else:
            print("Nepravilni opseg indeksa!")

        bottom_edge_right_x = x + w
        bottom_edge_right_y = y + h
        bottom_edge_right = (bottom_edge_right_x, bottom_edge_right_y)

        # Provera da li je yolo pravougaonik u okviru koji se posmatra:
        if 0 <= bottom_edge_right_x < hsv_image.shape[0] and 0 <= bottom_edge_right_y < hsv_image.shape[1]:
            hsv_bottom_edge_right = hsv_image[bottom_edge_right_x, bottom_edge_right_y]
        else:
            print("Nepravilni opseg indeksa!")

        # provere ivica detektovanog pravougaonika yolo detektovanog objekta
        if ((0 <= bottom_edge_right_x < hsv_image.shape[0] and 0 <= bottom_edge_right_y < hsv_image.shape[1]) and (0 <= bottom_edge_left_x < hsv_image.shape[0] and 0 <= bottom_edge_left_y < hsv_image.shape[1]) and
                (dodirujeBelu(hsv_bottom_edge_left) or ( (dodirujeBelu(hsv_bottom_center)) and (not dodirujePlavu(hsv_bottom_edge_left)) ) ) ):  ###
            on_white = True

        if ((0 <= bottom_edge_right_x < hsv_image.shape[0] and 0 <= bottom_edge_right_y < hsv_image.shape[1]) and (0 <= bottom_edge_left_x < hsv_image.shape[0] and 0 <= bottom_edge_left_y < hsv_image.shape[1]) and
                ((dodirujePlavu(hsv_bottom_edge_left) or (dodirujePlavu(hsv_bottom_center))) or (dodirujePlavu(hsv_bottom_edge_right) or (dodirujePlavu(hsv_bottom_center))))):  ###
            on_blue = True

        if (on_white and not on_blue): #da bi vratilo još frame-ova kada i yolo objekat ipak dodiruje put, sa not_on_blue, tj nije na putu se smanjuje sansa za pogresne detekcije
            cv.imwrite(f'naBelomIvicnjakuF1_{int(current_time)}.png', frame)
            frame[:100, :] = [0, 0, 255]  # crvena na vrhu prikaza

    cv.imshow('Track limits, turn 4', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


