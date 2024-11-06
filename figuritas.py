from djitellopy import Tello
import cv2
import time
import numpy as np

# Inicializar el dron y activar transmisión de video
tello = Tello()
tello.connect()
print(tello.get_battery())
tello.streamon()

# Tiempo límite sin detección (en segundos)
time_limit = 10
start_time = time.time()

def detectaFiguras():
    approx = []
    # Captura el frame de la cámara del dron
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (640, 480))  # Redimensionar para procesar más rápido
    frame = cv2.flip(frame, 0)

    frame = cv2.blur(frame, (14, 14))  # Tamaño del kernel (5, 5)

    # Convertir a escala de grises
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray Image", gray_image)

    # Aplicar binarización
    _, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Binarized Image", thresh_image)

    # Aplicar operaciones morfológicas para reducir ruido
    kernel = np.ones((5, 5), np.uint8)
    # thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("Cleaned Image", thresh_image)

    # Encontrar contornos
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Área mínima para filtrar el ruido (ajusta este valor según el tamaño de la figura)
    min_area = 3000
    # max_area = 0.9 * frame.shape[0] * frame.shape[1]
    maxArea = 100000

    # Iterar sobre cada contorno
    for i, contour in enumerate(contours):
        # Ignorar contornos que no cumplan con el área mínima
        area = cv2.contourArea(contour)
        if area < min_area or area > maxArea:
            continue

        # Aproximación de la forma
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Dibujar el contorno
        cv2.drawContours(frame, [approx], 0, (0, 0, 0), 4)

        # Coordenadas para etiquetar
        x, y, w, h = cv2.boundingRect(approx)
        x_mid = int(x + (w / 3))
        y_mid = int(y + (h / 1.5))
        coords = (x_mid, y_mid)
        colour = (0, 0, 0)
        font = cv2.FONT_HERSHEY_DUPLEX

        # Detectar y etiquetar la forma según el número de lados
        if len(approx) == 3:
            cv2.putText(frame, "Triangle", coords, font, 1, colour, 1)
        elif len(approx) == 4:
            cv2.putText(frame, "Quadrilateral", coords, font, 1, colour, 1)
        elif len(approx) == 5:
            cv2.putText(frame, "Pentagon", coords, font, 1, colour, 1)
        elif len(approx) == 10:
            cv2.putText(frame, "Star", coords, font, 1, colour, 1)
        #else:
         #   cv2.putText(frame, "Circle", coords, font, 1, colour, 1)

    # Mostrar el video con la figura detectada
    cv2.imshow("Dron Feed", frame)
    return approx

tello.takeoff()
print("Despega xd")
tello.send_rc_control(0,-15,0,0)
while True:
    
    lados = detectaFiguras()
    # Liberar recursos y cerrar la transmisión de video
    if len(lados) == 10:
        tello.land()
        print("Aterrizaje por estrella")
    
    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
tello.streamoff()
cv2.destroyAllWindows()