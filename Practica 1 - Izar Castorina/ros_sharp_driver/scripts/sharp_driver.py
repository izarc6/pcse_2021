#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Izar Castorina - Práctica 1 - PCSE

# La instrucción de 'coding' arriba es para que Python procese correctamente
# el file aunque haya carácteres non-ASCII

import rospy
import serial, time
import threading

global arduino

from sensor_msgs.msg import Range

# DONE: CREAR tres objetos Event, uno para cada thread de escritura por el puerto serie
write_sensor0 = threading.Event()
write_sensor1 = threading.Event()
write_sensor2 = threading.Event()

# Valores finales en metros para los tres sensores
range0 = 0.0
range1 = 0.0
range2 = 0.0

def requestMeasuresSensor(sensor,pub): # funcion de solicitud de dato del sensor y publicación en ROS

  print("DEBUG - requestMeasuresSensor para el sensor " + sensor + " en marcha")

  # creamos un mensaje tipo Range
  msg = Range()
  msg.header.frame_id = 'kobuki'  # Frame de referencia
  msg.radiation_type = 1          # Infrared
  msg.field_of_view = 0           
  msg.min_range = 0.2
  msg.max_range = 1.5

  while not rospy.is_shutdown(): # mientras no paremos el nodo

    arduino.write(sensor) # escribimos por el serie el número de sensor

    #print("Enviado " + sensor + " por el serial")

    if sensor == '0':

      # DONE: BLOQUEAR el thread con el objeto Event correspondiente
      write_sensor0.wait()

      msg.range = range0 # rellenamos el mensaje con el rango recibido

    elif sensor == '1':

      # DONE: BLOQUEAR el thread con el objeto Event correspondiente
      write_sensor1.wait()

      msg.range = range1 # rellenamos el mensaje con el rango recibido

    elif sensor == '2':

      # DONE: BLOQUEAR el thread con el objeto Event correspondiente
      write_sensor2.wait()

      msg.range = range2 # rellenamos el mensaje con el rango recibido

    # rellenamos la cabecera del mensaje con la hora actual
    msg.header.stamp = rospy.get_rostime()

    #publicamos el mensaje usando el "publisher" que nos han pasado por parámetro
    pub.publish(msg)

  return

def readSensors(): # función de recepción de los datos por el puerto serie

  print("DEBUG - readSensors en marcha")

  global range0
  global range1
  global range2

  while not rospy.is_shutdown(): # mientras no paremos el nodo

    buffer = arduino.read_until()

    #print("Buffer: " + str(buffer))

    if len(buffer) == 8: # R X : X X X CR LF <-- estructura de string en el buffer (CR = carriage return; LF = line feed, new line)
                         # 0 1 2 3 4 5  6  7 <-- número de byte (char)

      #EXTRAER el valor del rango recibido y ALMACENARLO en rangeM en metros
      rangeM = float(buffer[3]) + (float(buffer[4])/10) + (float(buffer[5])/100)
      #print("DEBUG - Read: " + str(rangeM) + " (Sensor " + buffer[1] + ")")

      if buffer[1] == '0':
        # Guardamos el valor recibido en la variable del sensor 0
        range0 = rangeM

        # DONE: LIBERAR el thread correspondiente al sensor 0
        write_sensor0.set()

      elif buffer[1] == '1':
        # Guardamos el valor recibido en la variable del sensor 1
        range1 = rangeM

        # DONE: LIBERAR el thread correspondiente al sensor 1
        write_sensor1.set()

      elif buffer[1] == '2':
        # Guardamos el valor recibido en la variable del sensor 2
        range2 = rangeM

        # DONE: LIBERAR el thread correspondiente al sensor 2
        write_sensor2.set()

  return

if __name__ == "__main__":

  # Inicializamos el nodo
  rospy.init_node('sharp_driver')

  # Leemos los parámetros
  port = rospy.get_param('~port', '/dev/ttyUSB0')
  rospy.loginfo('Port: %s', port)

  baud = 9600

  # Creamos tres "publishers", uno para cada sensor, para publicar mensajes del tipo "Range"
  pub0 = rospy.Publisher('~range0', Range, queue_size=1)
  pub1 = rospy.Publisher('~range1', Range, queue_size=1)
  pub2 = rospy.Publisher('~range2', Range, queue_size=1)

  # Inicializamos la conexión serie
  rospy.loginfo("Connecting to the device ...")
  try:
    arduino = serial.Serial(port, baud)
    time.sleep(2)
  except serial.SerialException:
    rospy.logfatal("It was not possible to connect to the device")
    exit(0)
  rospy.loginfo("Successfully connected to the device!")

  # DONE: CREAR tres threads que ejecuten la función "requestMeasuresSensor", pasando 2 parámetros:
                                                                  # (1) char con el número de sensor y
                                                                  # (2) el "publisher" correspondiente
  requestMeasuresSensor0 = threading.Thread(name="reqM0", target=requestMeasuresSensor, args=('0', pub0))
  requestMeasuresSensor1 = threading.Thread(name="reqM1", target=requestMeasuresSensor, args=('1', pub1))
  requestMeasuresSensor2 = threading.Thread(name="reqM2", target=requestMeasuresSensor, args=('2', pub2))

  # DONE: INICIAR los tres threads creados
  requestMeasuresSensor0.start()
  requestMeasuresSensor1.start()
  requestMeasuresSensor2.start()
  
  # DONE: CREAR un thread que ejecute la función "readSensors" e INICIARLO
  readSensorsThread = threading.Thread(name="readS", target=readSensors, args="")
  readSensorsThread.start()

  print('Todos los threads en marcha')

  # "spin" hasta que paremos el nodo.
  rospy.spin() # Los threads se estan ejecutando

  # DONE: ESPERAR a que acaben los cuatro threads
  requestMeasuresSensor0.join()
  requestMeasuresSensor1.join()
  requestMeasuresSensor2.join()
  readSensorsThread.join()

  # Cerramos la conexión serie
  arduino.close()

  print('All done')
  