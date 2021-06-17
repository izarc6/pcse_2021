# -*- coding: utf-8 -*-

from numpy.lib import math
import skimage.io, skimage.color, skimage.exposure, skimage.feature
import matplotlib.pyplot
import numpy
import time
from datetime import datetime

imgs = []

# Parámetros para el procesamiento
canny = [1, 0.2, 0.3]  # Valores para la detección de bordes

N = 4           # Número de lineas verticales para las imágenes (mínimo calculado empíricamente)
N_MAXIMOS = 8   # Número de máximos a tener en consideración
intervalo = 6   # Número de celdas alrededor del pico de Hough a limpiar

delta_theta = 0.5   # Intervalo para los grados
delta_rho = 3       # Intervalo para los píxeles

max_grados = 10 + 90    # Pi
min_grados = -10 + 90

# Parametros de debug
enable_stretched = True         # Habilita la multiplicación de las filas en el gráfico, para que no quede tan poco alto
stretching = 5                  # Factor de stretching (multiplicación vertical)
only_show_result = False        # Enseña solo los resultados finales, o sea las imagenes originales con lineas rojas superpuestas


def extraccion_contornos(img):
    global canny

    # Paso la imagen a escala de grises
    img_gray = skimage.color.rgb2gray(img)
    # Ecualizamos la imagen para mejorar el contraste
    # No quedan demasiado bien, pero lo que nos importa son los bordes
    img_str = skimage.exposure.rescale_intensity(img_gray)
    img_eq = skimage.exposure.equalize_hist(skimage.img_as_ubyte(img_str))

    # Detección de bordes por Canny según los parámetros especificados
    img_canny = skimage.feature.canny(img_eq, canny[0], canny[1], canny[2])
    #img_canny = skimage.feature.canny(img_eq)

    if not only_show_result:
        # Visualización resultado de Canny
        figure = matplotlib.pyplot.figure()
        skimage.io.imshow(img_canny.astype(numpy.uint8), cmap='gray')
        figure.show()

    #print("DEBUG - img_canny lens:" + str(len(img_canny)) + ", " + str(len(img_canny[0])))

    return img_canny

# Procesa el mapa de bordes, genera el acumulador de Hough según los parámetros globales
# y devuelve la imagen de resultado
def hough(img, img_contornos):
    global imgs, only_show_result, min_grados, max_grados, delta_theta, delta_rho, N_MAXIMOS

    hough = [] # Acumulador
    
    # Parametros de la imagen
    height = len(img_contornos)
    width = len(img_contornos[0])

    diagonal = numpy.sqrt(pow(height, 2) + pow(width, 2))
    print("Diametro de la imagen: " + str(diagonal))

    # Calcula las dimensiones del acumulador y lo llena de 0s
    p = (max_grados - min_grados)/delta_theta + 1
    q = (diagonal*2)/delta_rho + 1

    for i in range(int(p)):
        hough.append([])
        for j in range(int(q)):
            hough[i].append(0)

    print("Filas: " + str(len(hough)) + ", columnas: " + str(len(hough[0])))
    
    contador = 0    # Variable auxiliaria para el bucle

    theta_k = float(min_grados) # Valor incial de theta_k
    
    # Listas de strings auxiliarias
    errores=[]

    start = time.time_ns() # Para calcular el tiempo que se ha tardado en procesar la imagen
    
    # Algoritmo de generación del acumulador - basado en el pseudocódigo V2

    # Para cada pixel del mapa de contornos
    for u in range(height):
        for v in range(width):
            theta_k = min_grados
            if img_contornos[u][v] == False:    # Si no es un contorno, salta a la siguiente iteración
                continue
            contador += 1
            if contador % 100 == 0:
                print("Hough - {:.2f}%  ".format(u/height * 100), end="\r") # Visualiza el progreso en %
            
            # Recorre los valores de theta a intervalos de delta_theta
            while theta_k <= max_grados:
                rho = u * numpy.cos(theta_k * numpy.pi/180) + v * numpy.sin(theta_k * numpy.pi/180)
                rho_k = int(rho)

                # Calcula los índices dentro del acumulador
                s = int(((theta_k - min_grados) / delta_theta) + 1)
                t = int(((rho_k + diagonal) / delta_rho) + 1)

                # Intenta sumar 1 a la posición determinada, si hay errores no para la ejecución del programa
                try:
                    hough[s][t] += 1
                except:
                    errores.append("ERROR PARA LAS COORDENADAS u={:d}, v={:d} | theta_k={:2f}, s={:d} t={:d}, rho={:2f}, rho_k:{:2f}".format(u, v, theta_k, s, t, rho, rho_k))
                
                # Incrementa theta_k
                theta_k += delta_theta
            
            
    print("Hough - 100%  \r\nAcabado! Tiempo empleado: " + str((time.time_ns() - start)/1000000000) + " s.")
    print("Procesados " + str(contador) + " pixeles de borde con " + str(len(errores)) + " errores.")

    if not only_show_result:
        
        # Visualización del gráfico con stretching vertical
        if enable_stretched:
            hough_stretched = []

            for i in range(len(hough)):
                for j in range(stretching):
                    hough_stretched.append(hough[i])
    
        figure = matplotlib.pyplot.figure()
        if enable_stretched: skimage.io.imshow(numpy.array(hough_stretched))
        else: skimage.io.imshow(numpy.array(hough))
    
        figure.show()

    print("Encontrando máximos...")

    maximos = []

    hough_copia = hough
    
    for i in range(N_MAXIMOS):
        maximos.append(encontrar_maximo(hough_copia))
        hough_copia = limpiar_celdas(hough_copia, maximos[i][0], maximos[i][1])

    print("Máximos encontrados: " + str(len(maximos)))
    for elem in maximos: 
        print("\tu: {:d}\tv:{:d}\tValor: {:d}".format(elem[0], elem[1], elem[2]))

    # Visualización del gráfico después de eliminar los picos y sus vecinos
    if not only_show_result:
        if enable_stretched:    # Visualización con stretching vertical
            hough_stretched = []
            for i in range(len(hough)):
                for j in range(stretching):
                    hough_stretched.append(hough_copia[i])


        figure = matplotlib.pyplot.figure()
        if enable_stretched: skimage.io.imshow(numpy.array(hough_stretched))
        else: skimage.io.imshow(numpy.array(hough_copia))
        figure.show()

    v_horizontal = int(len(img_contornos) / 2)
    us = []

    # Encontramos las posiciones de las líneas verticales que hacen intersección
    # con la linea paralela al eje horizontal y posicionada a mitad de altura de la imagen

    # maximos[i][0] = s, maximos[i][1] = t, maximos[i][2] = valor de la celda
    for i in range(N):
        if i >= len(maximos): break  # Sal del bucle si i se pasa del número de máximos encontrados
        theta = (maximos[i][0] - 1) * delta_theta + min_grados - 90
        rho = (maximos[i][1] - 1) * delta_rho - diagonal
        print("Valores para pico {:d}, {:d} | Theta: {:.4f}\tRho: {:.4f}".format(maximos[i][0], maximos[i][1], theta, rho))
        u = (rho - (v_horizontal * numpy.sin(theta * numpy.pi/180))) / numpy.cos(theta * numpy.pi/180)
        us.append(u)

    print("Coordenadas X de intersección con el eje horizontal en mitad de la imagen encontradas:")
    print(us)

    # Visualiza los resultados
    figure = matplotlib.pyplot.figure()
    skimage.io.imshow(img)
    for u in us:
        matplotlib.pyplot.axvline(u, color="red")
    #matplotlib.pyplot.axhline(v_horizontal, color="blue")
    figure.show()
    
    # Devuelve las coordenadas X encontradas
    return us

def encontrar_maximo(matriz):
    max_actual = 0
    u_actual = 0
    v_actual = 0
    for u in range(len(matriz)):
        for v in range(len(matriz[0])):
            if matriz[u][v] > max_actual:
                max_actual = matriz[u][v]
                u_actual = u
                v_actual = v
    print("Encontrado máximo en u={:d}, v={:d}, valor={:d}".format(u, v, max_actual))
    return [u_actual, v_actual, max_actual]

def limpiar_celdas(matriz, u, v):
    global intervalo

    # Limpiamos los valores alrededor
    u_min = u - intervalo
    if u_min < 0: u_min = 0
    u_max = u + intervalo + 1
    if u_max > len(matriz): u_max = len(matriz)

    v_min = v - intervalo
    if v_min < 0: v_min = 0
    v_max = v + intervalo + 1
    if v_max > len(matriz[0]): v_max = len(matriz[0])

    #print("Limpiando celdas alrededor del máximo, u=[{:d} {:d}], v=[{:d} {:d}]".format(u_min, u_max, v_min, v_max))
    matriz[u][v] = 0
    for u_2 in range(u_min, u_max):
        for v_2 in range(v_min, v_max):
            matriz[u_2][v_2] = 0      
    
    return matriz
    
def trazaLineas(i):
    # Genera el mapa de contornos
    img_canny = extraccion_contornos(imgs[i-1])

    # Desde el mapa de bordes se calcula el acumulador de Hough
    hough(imgs[i-1], img_canny)



if __name__ == "__main__":

    print("*** Práctica 2 - Percepción ***\nAsignatura de Percepción y Control para Sistemas Empotrados\nIzar Castorina - Curso 2020/2021")
    print("Cargando imagenes", end="")
    for i in range(1, 6):
        # Carga las 5 imagenes en la lista imgs
        imgs.append(skimage.io.imread("image"+str(i)+".JPG"))
        print(".", end="")
    print(" Carga completada.")

    while True:
        print("\n***************\nOpciones")
        print("\t1-5\tEjecuta el script para la imagen correspondiente")
        print("\t6\tEjecuta el script para todas las imagenes de forma secuencial")
        print("\t9\tSal del programa")

        print("\ts\tHabilita/Deshabilita el stretching en la visualización de los acumuladores de Hough")
        print("\t\tEstado actual: ", end="")
        print("Activo") if enable_stretched else print("Desactivado")

        print("\tr\tHabilita/Deshabilita la visualización solamente de los resultados finales")
        print("\t\tEstado actual: ", end="")
        print("Activo") if only_show_result else print("Desactivado")

        print("")

        sel = input("Tu selección: ")

        if sel == "1" or sel == "2" or sel == "3" or sel == "4" or sel == "5":
            trazaLineas(int(sel))
        elif sel == "6":
            for i in range(1, 6):
                trazaLineas(i)
        elif sel == "9":
            print("Saliendo...")
            exit(0)

        elif sel == "s" or sel == "S":
            enable_stretched = not enable_stretched
        elif sel == "r" or sel == "R":
            only_show_result = not only_show_result

        
        else:
            print("Comando no reconocido.")
