import numpy as np
import time
import cv2 as cv
# Ovaj kod detektuje izlazak pre svega motocikala sa trkačke staze
# Konstante
whT = 320
confThreshold = 0.5
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
sensitivity = 1
line_thickness = 2
red_light_threshold = 0
leave_road_threshold = 0
# Kalmanov Filter to do
# Region of interest za detektovanje puta
roi_coordinates = [(0, 300), (640, 480)]
img_width, img_height = 640, 480
def pronadjiObjekte(outputs, image):
hT, wT, cT = image.shape
bounding_boxes = []
class_ids = []
confidences = []
target_classes = ["car", "motorbike", "bus", "truck"] #samo ove vrste yolo
detektovanih objekata će se obrađivati.
# Na početku sam imao dosta problema zbog pogrešne detekcije objekata koji ne
pripadju vozilima
for output in outputs:
for detection in output:
scores = detection[5:]
class_id = np.argmax(scores)
confidence = scores[class_id]
if confidence > confThreshold and classNames[class_id] in target_classes:
w, h = int(detection[2] * wT), int(detection[3] * hT)
x, y = int((detection[0] * wT) - w / 2), int((detection[1] * hT) - h / 2)
bounding_boxes.append([x, y, x + w, y + h]) # Append the bounding box as
a list
class_ids.append(class_id)
confidences.append(float(confidence))
#https://stackoverflow.com/questions/58621583/how-find-confidence-for-
each-classes-in-yolo-darknet
bounding_boxes = np.array(bounding_boxes)
indices = cv.dnn.NMSBoxes(bounding_boxes.tolist(), confidences, confThreshold,
nmsThreshold)
objects = []
for i in indices:
i = i[0] if isinstance(i, list) else i # Izvlačenje podataka iz liste, ako
postoje
x, y, x2, y2 = bounding_boxes[i]
w, h = x2 - x, y2 - y
objects.append((x, y, w, h, False, True))
cv.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2) # Zeleni pravougaonik
preko objekta
cv.putText(image, f'{classNames[class_ids[i]].upper()} {int(confidences[i] *
100)}%',
(x, y - 10), cv.FONT_ITALIC, 0.6, (0, 255, 0), 2) # Zeleni tekst
return objects
def tacka_blizu_puta(point, road_vertices):
# Provera da li je tačka blizu puta
pt = np.array(point, dtype=np.float32)
return cv.pointPolygonTest(road_vertices, tuple(pt), False) > 0
roi_size = 10 # Veličina ROI u pikselima
def nadjiPut(image, objects):
# Konvertovanje u crno belo
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# Postavljanje regiona od interesa
img_height, img_width = gray_image.shape[:2]
center_x, center_y = img_width // 2, img_height // 2
roi_vertices = np.array([
[(center_x - roi_size, center_y - roi_size), (center_x + roi_size, center_y -
roi_size),
(center_x + roi_size, center_y + roi_size), (center_x - roi_size, center_y +
roi_size)]],
dtype=np.int32)
# roi_vertices vrednost 0
maska_puta = np.zeros_like(gray_image)
maska_puta = cv.fillPoly(maska_puta, roi_vertices, 255)
# Hough transformacija za detektovanje belih linija na putu
linije = cv.HoughLinesP(maska_puta, 1, np.pi / 180, threshold=50, minLineLength=100,
maxLineGap=10)
#https://stackoverflow.com/questions/52816097/line-detection-with-opencv-python-and-
hough-transform
if linije is not None:
# Filtriranje da bi se izabrale samo one linije koje su blizu puta
linije_blizu_puta = []
for linija in linije:
x1, y1, x2, y2 = linija[0]
duzina_linije = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
# Prikaz samo onih linija koje su duže od neke unapred zadate vrednosti
if duzina_linije > 50:
pt1 = (x1, y1)
pt2 = (x2, y2)
# Dodaj liniju samo ako je blizu puta
if tacka_blizu_puta(pt1, roi_vertices[0]) or tacka_blizu_puta(pt2,
roi_vertices[0]):
linije_blizu_puta.append(linija)
for linija in linije_blizu_puta:
x1, y1, x2, y2 = linija[0]
cv.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2) # Obeležavanje linija
belom bojom na prikazu
# Filtriranje slike da bi se ponašao put
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
donji_opseg_sive = np.array([0, 0, 25], dtype=np.uint8)
gornji_opseg_sive = np.array([255, 25, 255], dtype=np.uint8)
gray_mask = cv.inRange(hsv_image, donji_opseg_sive, gornji_opseg_sive)
# Boja kojom obeležavam detektovan put, plava u ovom slučaju
boja_puta = (255, 0, 0)
image[gray_mask == 255] = boja_puta
# Iscrtavanje linija u ROI
ivice = cv.Canny(gray_image, 50, 150)
ivice_in_roi = cv.bitwise_and(ivice, maska_puta)
image[ivice_in_roi > 0] = (0, 255, 0) # Zelena boja za ivice u ROI
konture, _ = cv.findContours(maska_puta, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
konture_puta = [contour for contour in konture if cv.contourArea(contour) > 1000]
# Identifikacija bele linije sa jedne strane puta
for contour in konture_puta:
approximation = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True),
True)
if len(approximation) == 2:
cv.drawContours(image, [contour], -1, (255, 255, 255), 2) # Bela za belu
liniju oko puta
else:
cv.drawContours(image, [contour], -1, (128, 0, 128), 2) # Ljubičasta za
konture puta
return konture_puta, maska_puta
# isključivanje 10% ivica prikaza iz ROI.
edge_width = int(0.1 * img_width)
edge_height = int(0.1 * img_height)
roi_vertices = np.array([
[(edge_width, edge_height), (img_width - edge_width, edge_height),
(img_width - edge_width, img_height - edge_height), (edge_width, img_height -
edge_height)]],
dtype=np.int32)
granicni_objekti = []
def crtajPutILinije(frame, konturePuta, maska_puta, linije):
# Iscrtavanje kontura puta
cv.drawContours(frame, konturePuta, -1, (255, 255, 255), 2)
# Provera da li linije nisu None
if linije is not None:
# Iscrtavanje linija puta
for linija in linije:
x1, y1, x2, y2 = linija[0]
cv.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
def sivaBoja(color, threshold=10):
# np.abs (apsolutna vrednost) se koristi jer sam imao problem sa overflow issue
return np.all(np.abs(color - np.roll(color, 1)) < threshold)
def pronadjiIvicnjake(frame):
# Konverzija u HSV color space
hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
# Definicija boja za detekciju ivičnjaka
donjiOpsegIvicnjak = np.array([80, 40, 40], dtype=np.uint8)
gornjiOpsegIvicnjak = np.array([120, 255, 255], dtype=np.uint8)
# Filter na slici za bi ostala samo boja ivičnjaka
maskaIvicnjaka = cv.inRange(hsv_image, donjiOpsegIvicnjak, gornjiOpsegIvicnjak)
# Nalaženje kontura ivičnjaka
konture, _ = cv.findContours(maskaIvicnjaka, cv.RETR_EXTERNAL,
cv.CHAIN_APPROX_SIMPLE)
# Filteriranje kontura
kontureIvicnjaka = [contour for contour in konture if cv.contourArea(contour) > 100]
return kontureIvicnjaka
def crtanjeIvicnjaka(frame, konture, bojaIvicnjaka):
for contour in konture:
cv.drawContours(frame, [contour], -1, bojaIvicnjaka, 2)
cap = cv.VideoCapture('madjarskagranicestaze.mp4')
x = 0 #inicijalizacija
y = 0
w = 0
h = 0
while True:
current_time = time.time()
#https://stackoverflow.com/questions/71664323/timer-update-inside-while-loop-if-
statement
success, frame = cap.read()
if not success:
break
blob = cv.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
net.setInput(blob) #https://programtalk.com/python-more-
examples/cv2.cv2.dnn.blobFromImage/
layer_names = net.getLayerNames()
output_names = [(layer_names[i - 1]) for i in net.getUnconnectedOutLayers()]
outputs = net.forward(output_names)
#https://www.grepper.com/answers/526529/output_layers+%3D+%5Blayer_names%5Bi%5B0%5D+-
+1%5D+for+i+in+net.getUnconnectedOutLayers%28%29%5D+IndexError%3A+invalid+index+to+scalar
+variable.
#https: // www.computervision.zone /
projects /
objects = pronadjiObjekte(outputs, frame)
konturePuta, maska_puta = nadjiPut(frame, objects)
# Hough transformacija za linije puta ,
https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
linije = cv.HoughLinesP(maska_puta, 1, np.pi / 180, threshold=50, minLineLength=99,
maxLineGap=9)
# Iscrtavanje linija
crtajPutILinije(frame, konturePuta, maska_puta, linije)
# Iscrtavanje ivičnjaka
kontureIvicnjaka = pronadjiIvicnjake(frame)
crtanjeIvicnjaka(frame, kontureIvicnjaka, (0, 255, 255)) # Yellow color for curbs
# Ažuriranje graničnih objekata u granicniObjekti na osnovu toga da li objekat
preseca konture puta
for i, obj in enumerate(objects):
x, y, w, h, _, on_road = obj[:6]
bottom_center_x = x + w // 2
bottom_center_y = y + h
bottom_center = (bottom_center_x, bottom_center_y)
on_green_surface = False
for contour in konturePuta:
region_radius = 5
in_region = (
bottom_center_x - region_radius < contour[:, 0, 0].max() <
bottom_center_x + region_radius
and bottom_center_y - region_radius < contour[:, 0, 1].min() <
bottom_center_y + region_radius
)
if in_region:
on_road = True
break
# Provera da li donji delovi detektovanog vozila dodiruju zelenu površinu,
https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html
hsv_bottom_center = hsv_image[bottom_center_y, bottom_center_x]
if 40 <= hsv_bottom_center[0] <= 80 and 40 <= hsv_bottom_center[1] <= 255 and 40
<= hsv_bottom_center[2] <= 255:
on_green_surface = True
# Zapamti sliku kada objekat napusti put i dodirne zelenu površinu, detekcija za
motocikle
if (not on_road or on_green_surface):
cv.imwrite(f'slika_{int(current_time)}.png', frame)
objects[i] = obj[:4] + (on_road, on_green_surface)
granicniObjekti = [i for i, obj in enumerate(objects) if not obj[-1]]
for object_index in granicniObjekti:
x, y, w, h, _, _ = objects[object_index]
cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2) # Žut pravougaonik
ako je granični
# Crveno na vrhu ekrana kada imamo granicniObjekat (objekat koji dodiruje granicu
između površina)
if granicniObjekti:
frame[:50, :] = [0, 0, 255] # Crvena
else:
frame[:50, :] = [0, 0, 0] # Crna
# Detekcija ivica na putu
hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
# Opseg za sivu boju
donjiOpsegSive = np.array([10, 10, 30], dtype=np.uint8)
gornjiOpsegSive = np.array([189, 105, 189], dtype=np.uint8)
if h > 0 and w > 0:
# Koristi se posebna metoda daLiJeSiva, koja se oslanjanja na to da su
pojedinačni pikseli gotovo jednakih vrednosti kada je u pitanju siva boja
if sivaBoja(frame[y:y + h, x:x + w][0][0]):
bojaPuta = (128, 128, 128) # siva boja
else:
bojaPuta = (255, 0, 0) # Plava
# Filter za sivu boju
gray_mask = cv.inRange(hsv_image, donjiOpsegSive, gornjiOpsegSive)
frame[gray_mask == 255] = bojaPuta
cv.imshow('Detekcija izlaska sa staze', frame)
if cv.waitKey(1) & 0xFF == ord('q'):
break
cap.release()
cv.destroyAllWindows()
