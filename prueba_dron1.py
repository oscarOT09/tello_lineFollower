#Ejemplo para el movimiento
from djitellopy import tello
from time import sleep

#Conexión, creación de un objeto 
tell = tello.Tello()
tell.connect() #Se conecta xd

#Para saber cuanta pila tiene
print(tell.get_battery())

#Despegar
tell.takeoff()

#Controla la dirección (?) y las velocidades
#Velocidad left/rigth -100/100   
#Velocidad forward/backward -100/100
#Velocidad up/down -100/100
#Velocidad de rotación
tell.send_rc_control(0,0,40,0)
#Delay de 2 segundos
sleep(2)

#Aterriza
tell.land()