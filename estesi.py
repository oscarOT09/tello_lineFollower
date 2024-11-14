from djitellopy import tello
import cv2
import numpy as np
import time

# Configuración del dron Tello
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

# Configuración inicial
hsvVals = [20, 137, 172, 151, 205, 255]  # Valores de HSV para el seguimiento de línea
sensors = 3
threshold = 0.2
width, height = 480, 360
senstivity = 3
weights = [-20, -10, 0, 10, 20]
fSpeed = 15
takeoff = False

def thresholding(img):
    """ Aplica un filtro de color para segmentar el objeto """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def getContours(imgThres, img):
    """ Encuentra el centroide del contorno más grande y dibuja sobre la imagen """
    cx = 0
    contours, _ = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        cx = x + w // 2
        cy = y + h // 2
        cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    return cx

def getSensorOutput(imgThres, sensors):
    """ Divide la imagen en 'sensors' partes horizontales y devuelve el estado de cada parte """
    imgs = np.hsplit(imgThres, sensors)
    totalPixels = (imgThres.shape[1] // sensors) * imgThres.shape[0]
    senOut = []

    for x, im in enumerate(imgs):
        pixelCount = cv2.countNonZero(im)
        if pixelCount > threshold * totalPixels:
            senOut.append(1)
        else:
            senOut.append(0)
        cv2.imshow(f"Sensor {x}", im)
    return senOut

def sendCommands(senOut, cx):
    """ Envía comandos al dron basado en la salida de sensores y el centroide """
    global curve
    # Control de traslación
    lr = (cx - width // 2) // senstivity
    lr = int(np.clip(lr, -10, 10))
    if lr < 2 and lr > -2:
        lr = 0

    # Control de rotación según la salida de los sensores
    if senOut == [1, 0, 0]:
        curve = weights[0]
    elif senOut == [1, 1, 0]:
        curve = weights[1]
    elif senOut == [0, 1, 0]:
        curve = weights[2]
    elif senOut == [1, 1, 1]:
        curve = weights[3]
    elif senOut == [0, 0, 1]:
        curve = weights[4]
    else:
        curve = 0  # Mantén recto si no hay una señal clara

    me.send_rc_control(lr, fSpeed, 0, curve)

while True:
    if takeoff:
        img = me.get_frame_read().frame
        img = cv2.resize(img, (width, height))
        img = cv2.flip(img, 0)

        imgThres = thresholding(img)
        cx = getContours(imgThres, img)  # Obtiene el centroide para traslación
        senOut = getSensorOutput(imgThres, sensors)  # Salida de sensores 1x3
        sendCommands(senOut, cx)  # Envía comandos al dron

        cv2.imshow("Output", img)
        cv2.imshow("Thresholded", imgThres)
    else:
        # Muestra un mensaje para iniciar el despegue
        img_placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(img_placeholder, "Presiona 'E' para despegar", (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Output", img_placeholder)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('e') and not takeoff:
        me.takeoff()
        time.sleep(1)
        takeoff = True
    elif key == ord('q'):
        me.land()
        break

me.streamoff()
cv2.destroyAllWindows()