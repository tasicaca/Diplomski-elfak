import numpy as np
import time
import cv2 as cv
# Ovaj kod detektuje izlazak vozila sa trkačke staze
#1.1.2024. dodato poboljsanje za detekciju silaska na zelenu i tirkiznu povrsinu+
#to do, da prikazem ONO ŠTO STVARNO DETEKTUJE, TJ NA OSNOVU ČEGA PRAVI SCREENSHOTOVE
#dodate ivice , prikaz detektovanih ivica
#OK RADI SADA , NE BAGUJE PRIKAZ NA POLA

# Konstante
whT = 320
confThreshold = 0.3
nmsThreshold = 0.3

# YOLO MODEL
classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
modelConfig = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
# Definisanje line_coordinates
line_coordinates = [(100, 300), (500, 300)]
sensitivity = 1
line_thickness = 2
red_light_threshold = 0
leave_road_threshold = 0

# Region of interest za detektovanje puta
roi_coordinates = [(0, 300), (640, 480)]
img_width, img_height = 660, 660

#def sivaBoja(pixel):
#    range_value = 100
#    #np.abs (apsolutna vrednost) se koristi jer sam imao problem sa overflow issue
#    return all(np.abs(np.diff(pixel)) <= range_value)

def sivaBoja(color):
    threshold = 150
    channel_diff = np.abs(np.diff(color))
    return np.all(channel_diff < threshold)

def tacka_blizu_puta(point, road_vertices):
    # Provera da li je tačka blizu puta
    pt = np.array(point, dtype=np.float32)
    return cv.pointPolygonTest(road_vertices, tuple(pt), False) > 0

roi_size = 10  # Veličina ROI u pikselima

def pronadjiput(slika):   #KORISTI SE DA OZNACI POVRSINU PUTA              #funkcijaZaF1kod, iz koda 26122023-1501ZamotogpdetektujeSilazakNazelenePovrsineModifikovanZaFormulu
    # Convert the frame to grayscale
    siva_slika = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)

    # adaptive thresholding https://pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/
    adaptive_threshold = cv.adaptiveThreshold(siva_slika, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2) #THRESH_BINARY_INV

    # Region koji predstavlja povrsinu puta
    maska_puta = adaptive_threshold.copy()

    # (ROI)
    roi_vertices = np.array([[(450, 150), (450, 150), (450, 750), (450, 750)]], dtype=np.int32)
    maska_puta = cv.fillPoly(maska_puta, roi_vertices, 0)

    # Hafova transformacija linija za detekciju ivica puta
    linije = cv.HoughLinesP(maska_puta, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=10)
    ###maska puta i detekcija po opsegu sive boje mi radi, to sam proverio, sa Hough transformacijom u nekim slučajevima ima problema
    if linije is not None:
        # Filtriranje linija da se zadrže samo one blizu puta
        blizu_puta = []
        for linija in linije:
            x1, y1, x2, y2 = linija[0]
            duzina = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Prikazivanje samo linija koje su dovoljno duge
            if duzina > 20:
                # Konvertovanje koordinata u pre nego što se proslede na pointPolygonTest
                pt1 = (x1, y1)
                pt2 = (x2, y2)
                # Dodaje se linija samo ako je blizu puta
                if tacka_blizu_puta(pt1, roi_vertices[0]) or tacka_blizu_puta(pt2, roi_vertices[0]):
                    blizu_puta.append(linija)

        for linija in blizu_puta:
            x1, y1, x2, y2 = linija[0]
            cv.line(slika, (x1, y1), (x2, y2), (255, 0, 0), 1)  # plava boja za liniju

    # Filtriranje slike da se dobije boja kojom se dodatno detektuje put
    hsv_slika = cv.cvtColor(slika, cv.COLOR_BGR2HSV)
    donja_siva = np.array([5, 5, 60], dtype=np.uint8)
    gornja_siva = np.array([125, 60, 125], dtype=np.uint8)

    maska_sive = cv.inRange(hsv_slika, donja_siva, gornja_siva)

    """for i in range(slika.shape[0]):
        for j in range(slika.shape[1]):
            if not sivaBoja(slika[i, j]):
                maska_sive[i, j] = 0
    """
    combined_mask = cv.bitwise_and(maska_sive, maska_puta) #rezultat se dobija and operacijom nad maskom_sive i maskom_puta

    # Proizvoljna boja za obelezavanje puta (npr. plava)
    boja_puta = (255, 0, 0)

    # Oznacavanje puta na frejmu
    frame[combined_mask == 255] = boja_puta

    konture, _ = cv.findContours(maska_puta, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    konture_puta = [kontura for kontura in konture if cv.contourArea(kontura) > 1000]

    # Identifikacija bele linije s jedne strane puta
    for kontura in konture_puta:
        aproksimacija = cv.approxPolyDP(kontura, 0.02 * cv.arcLength(kontura, True), True)
        if len(aproksimacija) >= 100:
            cv.drawContours(slika, [kontura], -1, (255, 0, 0), 1)  # plava boja za ivice puta
        #else:
        #    cv.drawContours(slika, [kontura], -1, (255, 255, 255), 1)  # bela boja za kraće linije

    return combined_mask     #konture_puta, combined_mask

cap = cv.VideoCapture('granicestazeKatar.mp4') ###########################adjarskagranicestaze1##############################################MENJA SE ULAZNI VIDEO
x = 0 #inicijalizacija
y = 0
w = 0
h = 0

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('outputSaPlavomBojom.mp4', fourcc, 20.0, (width, height)) ###posto sam promenio da put bude plav 100%

while True:
    current_time = time.time()
    #https://stackoverflow.com/questions/71664323/timer-update-inside-while-loop-if-statement
    success, frame = cap.read()
    if not success:
        break

    blob = cv.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)    #https://programtalk.com/python-more-examples/cv2.cv2.dnn.blobFromImage/
    layer_names = net.getLayerNames()
    output_names = [(layer_names[i - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_names)       #https://www.grepper.com/answers/526529/output_layers+%3D+%5Blayer_names%5Bi%5B0%5D+-+1%5D+for+i+in+net.getUnconnectedOutLayers%28%29%5D+IndexError%3A+invalid+index+to+scalar+variable.
                                            #https: // www.computervision.zone / projects /
    #objects = pronadjiObjekte(outputs, frame)# Ne koristi se detekcija u ovom kodu

    maska_puta = pronadjiput(frame)   #izmena da bi se prikazale ivice kako sam zeleo

    # Display the result frame
    out.write(frame)
    cv.imshow('Prikaz puta', frame)

    #cv.imshow('Detekcija izlaska sa staze', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

