import numpy as np
import time
import cv2 as cv

# Ovaj kod detektuje izlazak vozila sa trkačke staze
# 1.1.2024. dodato poboljsanje za detekciju silaska na zelenu i tirkiznu povrsinu+
# to do, da prikazem ONO ŠTO STVARNO DETEKTUJE, TJ NA OSNOVU ČEGA PRAVI SCREENSHOTOVE
# dodate ivice , prikaz detektovanih ivica
# OK RADI SADA , NE BAGUJE PRIKAZ NA POLA

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

roi_coordinates = [(0, 0), (660, 660)]
img_width, img_height = 660, 660

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


def tacka_blizu_puta(point, road_vertices):
    # Provera da li je tačka blizu puta
    pt = np.array(point, dtype=np.float32)
    return cv.pointPolygonTest(road_vertices, tuple(pt), False) > 0


roi_size = 10  # Veličina ROI u pikselima


# isključivanje 10% ivica prikaza iz ROI.
edge_width = int(0.1 * img_width)
edge_height = int(0.1 * img_height)
roi_vertices = np.array([
    [(edge_width, edge_height), (img_width - edge_width, edge_height),
     (img_width - edge_width, img_height - edge_height), (edge_width, img_height - edge_height)]],
    dtype=np.int32)

granicni_objekti = []


def crtajPutILinije(frame, konturePuta, maska_puta, linije):
    # Iscrtavanje kontura puta
    # Provera da li linije nisu None
    if linije is not None:
        # Iscrtavanje linija puta
        for linija in linije:
            x1, y1, x2, y2 = linija[0]
            cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv.drawContours(frame, konturePuta, -1, (255, 255, 0), 1)


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
    konture, _ = cv.findContours(maskaIvicnjaka, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filteriranje kontura
    kontureIvicnjaka = [contour for contour in konture if cv.contourArea(contour) > 100]

    return kontureIvicnjaka


def crtanjeIvicnjaka(frame, konture, bojaIvicnjaka):
    for contour in konture:
        cv.drawContours(frame, [contour], -1, bojaIvicnjaka, 2)


def adaptive_color_threshold(frame, center_x, center_y):
    # Calculate adaptive HSV ranges based on the current frame
    region_radius = 5
    region = frame[center_y - region_radius:center_y + region_radius, center_x - region_radius:center_x + region_radius]
    hsv_region = cv.cvtColor(region, cv.COLOR_BGR2HSV)

    hsv_lower = np.min(hsv_region, axis=(0, 1)) - 20
    hsv_upper = np.max(hsv_region, axis=(0, 1)) + 20

    return hsv_lower, hsv_upper


def enhance_detection(frame):
    # Convert the frame from BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green color in HSV
    lower_green = np.array([80, 205, 150])
    upper_green = np.array([130, 250, 215])

    # Define the lower and upper bounds for turquoise color in HSV
    lower_turquoise = np.array([80, 205, 150])
    upper_turquoise = np.array([130, 250, 215])

    # Threshold the frame to get binary masks for green and turquoise colors
    mask_green = cv.inRange(hsv, lower_green, upper_green)
    mask_turquoise = cv.inRange(hsv, lower_turquoise, upper_turquoise)

    # Combine the masks to get a final mask for green and turquoise surfaces
    final_mask = cv.bitwise_or(mask_green, mask_turquoise)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv.morphologyEx(final_mask, cv.MORPH_OPEN, kernel)
    final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Paint detected areas with clear turquoise color
    result_frame = frame.copy()
    clear_turquoise = (222, 222, 0)  # RGB values for clear turquoise color

    for cnt in contours:
        cv.drawContours(result_frame, [cnt], -1, clear_turquoise, -1)  # -1 fills the contour

    return result_frame


# adaptive color thresholding mi ne radi dobro.

def is_turquoise_or_green_not_blue(hsv_pixel):
    """
    Check if the given HSV pixel corresponds to turquoise or green color but not blue.
    """
    # Define HSV range for turquoise color
    turquoise_hsv_range = [(80, 110), (100, 255), (100, 255)]

    # Define HSV range for green color
    green_hsv_range = [(40, 80), (40, 255), (40, 255)]

    # Define HSV range for blue color
    blue_hsv_range = [(90, 130), (50, 255), (50, 255)]

    # Check if the pixel is within the range for either turquoise or green but not blue
    is_turquoise = all(H_MIN <= hsv_pixel[i] <= H_MAX for i, (H_MIN, H_MAX) in enumerate(turquoise_hsv_range))
    is_green = all(H_MIN <= hsv_pixel[i] <= H_MAX for i, (H_MIN, H_MAX) in enumerate(green_hsv_range))
    is_blue = all(H_MIN <= hsv_pixel[i] <= H_MAX for i, (H_MIN, H_MAX) in enumerate(blue_hsv_range))

    return (is_turquoise or is_green) and not is_blue


# Source: OpenCV Canny Edge Detection,Source: OpenCV Contours
def touching_blue(hsv_pixel):

    blue_hsv_range = [(240, 240), (240, 240), (255, 255)]

    is_blue = all(H_MIN <= hsv_pixel[i] <= H_MAX for i, (H_MIN, H_MAX) in enumerate(blue_hsv_range))

    return is_blue


def pronadjiput(slika,objekti):  # funkcijaZaF1kod, iz koda 26122023-1501ZamotogpdetektujeSilazakNazelenePovrsineModifikovanZaFormulu
    # Convert the frame to grayscale
    siva_slika = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)

    # Use adaptive thresholding https://pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/
    adaptive_threshold = cv.adaptiveThreshold(siva_slika, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

    # Set the region inside the adaptive threshold as the road
    maska_puta = adaptive_threshold.copy()

    # Use region of interest (ROI) for road detection
    roi_vertices = np.array([[(450, 150), (450, 150), (450, 750), (450, 750)]], dtype=np.int32)
    maska_puta = cv.fillPoly(maska_puta, roi_vertices, 0)

    # Hafova transformacija linija za detekciju bele linije puta
    linije = cv.HoughLinesP(maska_puta, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

    if linije is not None:
        # Filtriranje linija da se zadrže samo one blizu puta
        blizu_puta = []
        for linija in linije:
            x1, y1, x2, y2 = linija[0]
            duzina = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Prikazivanje samo linija koje su dovoljno duge (prilagoditi dužinu prema potrebi)
            if duzina > 200:
                # Konvertuj koordinate u tuple pre nego što ih prosledite pointPolygonTest
                pt1 = (x1, y1)
                pt2 = (x2, y2)
                # Dodajte liniju samo ako je blizu puta
                if tacka_blizu_puta(pt1, roi_vertices[0]) or tacka_blizu_puta(pt2, roi_vertices[0]):
                    blizu_puta.append(linija)

        for linija in blizu_puta:
            x1, y1, x2, y2 = linija[0]
            cv.line(slika, (x1, y1), (x2, y2), (255, 255, 0), 10)  # White color for the line

    # Filtriranje slike da se dobije boja kojom se dodatno detektuje put
    hsv_slika = cv.cvtColor(slika, cv.COLOR_BGR2HSV)
    donja_siva = np.array([10, 10, 30], dtype=np.uint8)
    gornja_siva = np.array([189, 90, 189], dtype=np.uint8)

    maska_sive = cv.inRange(hsv_slika, donja_siva, gornja_siva)

    combined_mask = cv.bitwise_and(maska_sive, maska_puta)

    # Neka boja za obelezavanje puta (npr. plava)
    boja_puta = (255, 0, 0)

    # Oznaci put na frejmu
    frame[combined_mask == 255] = boja_puta
    cv.imshow('Enhanced Detection', frame)

    konture, _ = cv.findContours(maska_puta, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    konture_puta = [kontura for kontura in konture if cv.contourArea(kontura) > 1000]

    # Identifikacija bele linije s jedne strane puta
    for kontura in konture_puta:
        aproksimacija = cv.approxPolyDP(kontura, 0.02 * cv.arcLength(kontura, True), True)
        if len(aproksimacija) >= 2:
            cv.drawContours(slika, [kontura], -1, (255, 0, 0), 1)  # white color for road edges
        else:
            cv.drawContours(slika, [kontura], -1, (128, 0, 128), 1)  # purple color for the line

    return konture_puta, combined_mask


cap = cv.VideoCapture('outputSaPlavomBojom.mp4')
x = 0  # inicijalizacija
y = 0
w = 0
h = 0

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('outputSaPlavomBojom2.mp4', fourcc, 20.0, (width, height))

while True:
    current_time = time.time()
    # https://stackoverflow.com/questions/71664323/timer-update-inside-while-loop-if-statement
    success, frame = cap.read()
    if not success:
        break

    hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Gaussian Blur za smanjenje šuma
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Canny detektor ivica https://pyimagesearch.com/2021/05/12/opencv-edge-detection-cv2-canny/
    edges = cv.Canny(blurred, 50, 150)

    # cv.imshow('Video sa ivicama', edges)
    #############################################################

    blob = cv.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)  # https://programtalk.com/python-more-examples/cv2.cv2.dnn.blobFromImage/
    layer_names = net.getLayerNames()
    output_names = [(layer_names[i - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_names)  # https://www.grepper.com/answers/526529/output_layers+%3D+%5Blayer_names%5Bi%5B0%5D+-+1%5D+for+i+in+net.getUnconnectedOutLayers%28%29%5D+IndexError%3A+invalid+index+to+scalar+variable.
                                         # https: // www.computervision.zone / projects /

    objects = pronadjiObjekte(outputs, frame)

    konturePuta, maska_puta = pronadjiput(frame, objects)  # izmena da bi se prikazale ivice kako sam zeleo

    not_on_blue = False
    for i, obj in enumerate(objects):
        x, y, w, h, _, on_road = obj[:6]
        bottom_center_x = x + w // 2
        bottom_center_y = y + h
        bottom_center = (bottom_center_x, bottom_center_y)

        if 0 <= bottom_center_x < hsv_image.shape[0] and 0 <= bottom_center_y < hsv_image.shape[1]:
            hsv_bottom_center = hsv_image[bottom_center_x, bottom_center_y]
        else:
            print("Problem sa opsegom!")

        bottom_edge_left_x = x
        bottom_edge_left_y = y + h
        bottom_edge_left = (bottom_edge_left_x, bottom_edge_left_y)

        # Boundary Checks: Before accessing the pixel value, you can add a boundary check to ensure that the indices are valid:
        if (0 <= bottom_edge_left_x < hsv_image.shape[0] and 0 <= bottom_edge_left_y < hsv_image.shape[1]):
            hsv_bottom_edge_left = hsv_image[bottom_edge_left_x, bottom_edge_left_y]
        else:
            print("Problem sa opsegom!")

        bottom_edge_right_x = x + w
        bottom_edge_right_y = y + h
        bottom_edge_right = (bottom_edge_right_x, bottom_edge_right_y)

        # Boundary Checks: Before accessing the pixel value, you can add a boundary check to ensure that the indices are valid:
        if 0 <= bottom_edge_right_x < hsv_image.shape[0] and 0 <= bottom_edge_right_y < hsv_image.shape[1]:
            hsv_bottom_edge_right = hsv_image[bottom_edge_right_x, bottom_edge_right_y]
        else:
            print("Problem sa opsegom!")

        # mora neka ivica da ne dodiruje povrsinu puta
        if (0 <= bottom_edge_right_x < hsv_image.shape[0] and 0 <= bottom_edge_right_y < hsv_image.shape[1]) and (0 <= bottom_edge_left_x < hsv_image.shape[0] and 0 <= bottom_edge_left_y < hsv_image.shape[1]) and ((not touching_blue(hsv_bottom_edge_left) and (not touching_blue(hsv_bottom_center))) and (not touching_blue(hsv_bottom_edge_right) and (not touching_blue(hsv_bottom_center)))):
            not_on_blue = True

        on_green_surface = False

        for contour in konturePuta:
            region_radius = 5
            in_region = (
                    bottom_center_x - region_radius < contour[:, 0, 0].max() < bottom_center_x + region_radius
                    and bottom_center_y - region_radius < contour[:, 0, 1].min() < bottom_center_y + region_radius
            )

            if in_region:
                on_road = True
                break

        H_MIN, H_MAX = 40, 110
        S_MIN, S_MAX = 40, 255
        V_MIN, V_MAX = 40, 255

        # Provera da li donji delovi detektovanog vozila dodiruju zelenu površinu, https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html
        # hsv_bottom_center = hsv_image[bottom_center_y, bottom_center_x]
        # if H_MIN <= hsv_bottom_center[0] <= H_MAX and S_MIN <= hsv_bottom_center[1] <= S_MAX and V_MIN <= hsv_bottom_center[2] <= V_MAX:
        # on_green_surface = True

        """if 0 <= bottom_center_y < hsv_image.shape[0] and 0 <= bottom_center_x < hsv_image.shape[1]:
            hsv_bottom_center = hsv_image[bottom_center_y, bottom_center_x]
        else:
            print("Invalid indices")
        hsv_bottom_center = hsv_image[bottom_center_y, bottom_center_x]
        if is_turquoise_or_green_not_blue(hsv_bottom_center):
            on_green_surface = True
        """
        # hsv_bottom_center = hsv_image[bottom_center_y, bottom_center_x]
        # hsv_lower, hsv_upper = adaptive_color_threshold(frame, bottom_center_x, bottom_center_y)

        # if (hsv_lower[0] <= hsv_bottom_center[0] <= hsv_upper[0] and hsv_lower[1] <= hsv_bottom_center[1] <= hsv_upper[1] and hsv_lower[2] <= hsv_bottom_center[2] <= hsv_upper[2]):
        #    on_green_surface = True

        # Ukoliko nije na plavoj povrsini:
        if (not_on_blue):
            cv.imwrite(f'slikamotoGP_{int(current_time)}.png', frame)

        objects[i] = obj[:4] + (on_road, on_green_surface)

    granicniObjekti = [i for i, obj in enumerate(objects) if not obj[-1]]

    for object_index in granicniObjekti:
        x, y, w, h, _, _ = objects[object_index]

        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)  # Žut pravougaonik ako je granični

    # Crveno na vrhu ekrana kada imamo granicniObjekat (objekat koji dodiruje granicu između površina)
    if granicniObjekti:
        frame[:50, :] = [0, 0, 255]  # Crvena
    else:
        frame[:50, :] = [0, 0, 0]  # Crna

    out.write(frame)
    cv.imshow('Enhanced Detection', frame)

    # cv.imshow('Detekcija izlaska sa staze', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
