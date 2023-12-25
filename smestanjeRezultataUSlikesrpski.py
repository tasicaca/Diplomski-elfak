import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture('madjarskagranicestaze.mp4')
whT = 320
confPrag = 0.5
nmsPrag = 0.3

# UČITAJ YOLO MODEL
classesFile = 'coco.names'
naziviKlasa = []
with open(classesFile, 'rt') as f:
    naziviKlasa = f.read().rstrip('\n').split('\n')

konfiguracijaModela = "yolov3.cfg"
tezineModela = "yolov3.weights"
net = cv.dnn.readNetFromDarknet(konfiguracijaModela, tezineModela)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Definiši koordinate linije i osetljivost (prilagodi po potrebi)
koordinateLinije = [(100, 300), (500, 300)]
osetljivost = 1
debljinaLinije = 2

# Definiši prag vremena za prikaz crvenog svetla
pragCrvenoSvetlo = 0  # u sekundama

# Prilagodi osetljivost za detekciju objekata koji napuštaju put
pragNapustiPut = 0  # u sekundama

def pronadji_objekte(izlazi, slika):
    visina, sirina, brojKanala = slika.shape
    okviri = []
    idKlasa = []
    poverenja = []

    for izlaz in izlazi:
        for detekcija in izlaz:
            skorovi = detekcija[5:]
            idKlase = np.argmax(skorovi)
            poverenje = skorovi[idKlase]

            if poverenje > confPrag:
                sirina, visina = int(detekcija[2] * sirina), int(detekcija[3] * visina)
                x, y = int((detekcija[0] * sirina) - sirina / 2), int((detekcija[1] * visina) - visina / 2)
                okviri.append([x, y, sirina, visina])
                idKlasa.append(idKlase)
                poverenja.append(float(poverenje))

    indeksi = cv.dnn.NMSBoxes(okviri, poverenja, confPrag, nmsPrag)

    objekti = []
    for i in indeksi:
        okvir = okviri[i]
        x, y, sirina, visina = okvir[0], okvir[1], okvir[2], okvir[3]
        objekti.append((x, y, sirina, visina, True))
        cv.rectangle(slika, (x, y), (x + sirina, y + visina), (0, 255, 0), 2)  # Zelena pravougaonik
        cv.putText(slika, f'{naziviKlasa[idKlasa[i]].upper()} {int(poverenja[i] * 100)}%',
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Zeleni tekst

    return objekti

def pronadji_put(slika, objekti):
    # Konvertuj sliku u sivu
    siva_slika = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)

    # Koristi Canny detektor ivica
    ivice = cv.Canny(siva_slika, 50, 150)

    # Kreiraj praznu masku
    maska_puta = np.zeros_like(siva_slika)

    # Postavi region unutar detektovanih ivica kao put
    maska_puta[ivice > 0] = 255

    konture, _ = cv.findContours(maska_puta, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    konture_puta = [kontura for kontura in konture if cv.contourArea(kontura) > 1000]

    # Identifikacija bele linije s jedne strane puta
    for kontura in konture_puta:
        aproksimacija = cv.approxPolyDP(kontura, 0.02 * cv.arcLength(kontura, True), True)
        if len(aproksimacija) == 2:
            cv.drawContours(slika, [kontura], -1, (255, 255, 255), 2)  # Bela boja za liniju
        else:
            cv.drawContours(slika, [kontura], -1, (128, 0, 128), 2)  # Ljubičasta boja za ivice puta

    return konture_puta, maska_puta

prelazeci_objekti = []

while True:
    trenutno_vreme = time.time()  # Dobij trenutno vreme

    uspeh, frejm = cap.read()
    if not uspeh:
        break

    blob = cv.dnn.blobFromImage(frejm, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    imenaSlojeva = net.getLayerNames()
    izlaznaImena = [(imenaSlojeva[i - 1]) for i in net.getUnconnectedOutLayers()]
    izlazi = net.forward(izlaznaImena)

    objekti = pronadji_objekte(izlazi, frejm)
    konture_puta, maska_puta = pronadji_put(frejm, objekti)

    # Ažuriraj prelazeci_objekti na osnovu toga da li se objekat seče sa konturama puta
    for i, obj in enumerate(objekti):
        x, y, sirina, visina, _ = obj
        centar_objekta = (x + sirina // 2, y + visina // 2)

        # Povećaj strogoću za detekciju napuštanja puta
        na_putu = False
        for kontura in konture_puta:
            if cv.pointPolygonTest(kontura, centar_objekta, False) >= 0:
                na_putu = True
                break

        # Sačuvaj frejm kada objekat napusti put
        if not na_putu:
            cv.imwrite(f'frejm_{int(trenutno_vreme)}.png', frejm)

        objekti[i] = obj[:4] + (na_putu,)

    prelazeci_objekti = [i for i, obj in enumerate(objekti) if not obj[-1]]

    for indeks_objekta in prelazeci_objekti:
        x, y, sirina, visina, _ = objekti[indeks_objekta]
        cv.rectangle(frejm, (x, y), (x + sirina, y + visina), (0, 255, 255), 2)  # Žuti pravougaonici

    # Prikaz crvene trake na vrhu ekrana kada se detektuju objekti izvan puta
    if prelazeci_objekti:
        frejm[:40, :] = [0, 0, 255]  # Crvena boja https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
    else:
        frejm[:40, :] = [0, 0, 0]  # Postavi na crno ako nema detektovanih objekata ovo 40 je koliko piksela je bar

    # Detekcija ivica puta i obeležavanje puta bojom
    hsv_slika = cv.cvtColor(frejm, cv.COLOR_BGR2HSV)

    # Definiši opseg boja za detekciju sive boje puta
    donja_siva = np.array([0, 0, 20], dtype=np.uint8)
    gornja_siva = np.array([189, 85, 159], dtype=np.uint8)

    # Filtriranje slike da se dobije samo siva boja puta
    maska_sive = cv.inRange(hsv_slika, donja_siva, gornja_siva)

    # Neka boja za obeležavanje puta (npr. plava)
    boja_puta = (255, 0, 0)

    # Označi put na frejmu
    frejm[maska_sive == 255] = boja_puta

    cv.imshow('Frejm', frejm)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
#https://www.computervision.zone/topic/download-files-4/
