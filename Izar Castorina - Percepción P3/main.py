# -*- coding: utf-8 -*-

# Izar Castorina - Grupo 1

# RGB -> HSV
# H: Average
# S: Average
# V: Average

# RGB -> Greyscale
# E: GLCM
# H: GLCM

import sys
from numpy.lib import math
import skimage.io, skimage.color, skimage.exposure, skimage.feature, skimage.util
import matplotlib
import matplotlib.pyplot
import numpy
import time
from datetime import datetime
import random
from mpl_toolkits import mplot3d
from collections import Counter

imgs = []


# Parámetros para el procesamiento
canny = [1, 0.2, 0.3]  # Valores para la detección de bordes

perc_rho = 1.4
perc_nit = 1000

saved_data = None # Tabla de datos
saved_ws = None   # Resultado del Perceptrón

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
only_show_result = True         # Enseña solo los resultados finales, o sea las imagenes originales con lineas rojas superpuestas



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
    print("Procesados " + str(contador) + " pixeles de borde.") # con " + str(len(errores)) + " errores.")  # shhhhhh
    
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
        
    print("Coordenadas X de intersección con el eje horizontal en mitad de la imagen encontradas: " + str(us))
    
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
    #print("\tEncontrado máximo en u={:d}, v={:d}, valor={:d}".format(u_actual, v_actual, max_actual))
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
    lineas = hough(imgs[i-1], img_canny)

    return lineas
    
    
### ^^ Practica 2
##############################
### vv Practica 3
    
def obtenerCaracteristicas(forced):

    global saved_data

    if forced or saved_data is None:

        # Hue, saturación y energía
        
        data = []   # Tabla de datos de aprendizaje
        
        patches_d = []
        patches_w = []
        
        # Carga los patches de entreno
        # Las cantidades son fijas ahora mismo, pero se podría reemplazar con: cwd = os.getcwd(); arr = os.listdir(cwd + "/patches/door o wall"); n_files = len(arr)
        for i in range(1, 492):
            patches_d.append(skimage.io.imread("patches/door/patchD_" + str(i) + ".JPG"))
        
        for i in range(1, 455):
            patches_w.append(skimage.io.imread("patches/wall/patchW_" + str(i) + ".JPG"))
        
        # Para cada patch, calcula tono, saturación y energía, y añade los valores a la tabla 'data'
        for i in range(len(patches_d)):
            patch_hsv = skimage.color.rgb2hsv(patches_d[i])
            hue = numpy.average(patch_hsv[0])
            saturation = numpy.average(patch_hsv[1])
            
            glcm = skimage.feature.greycomatrix(skimage.img_as_ubyte(skimage.color.rgb2gray(patches_d[i])), distances=[1], angles=[0], levels=256, normed=True)
            energy = skimage.feature.greycoprops(glcm, prop='energy')
            
            data.append(["door_patch_"+str(i+1), hue, saturation, energy[0][0], 1])
        
        for i in range(len(patches_w)):
            patch_hsv = skimage.color.rgb2hsv(patches_w[i])
            hue = numpy.average(patch_hsv[0])
            saturation = numpy.average(patch_hsv[1])
            
            glcm = skimage.feature.greycomatrix(skimage.img_as_ubyte(skimage.color.rgb2gray(patches_w[i])), distances=[1], angles=[0], levels=256, normed=True)
            energy = skimage.feature.greycoprops(glcm, prop='energy')
            
            data.append(["wall_patch_"+str(i+1), hue, saturation, energy[0][0], 2])
            
        print("Total patches | {:d} door, {:d} wall".format(len(patches_d), len(patches_w)))    
        
        """
        print("Nombre\t\t\tHue\t\t\t\t\tSaturación\t\t\tEnergía\t\t\t\tClass")
        for elem in data:
            print(elem[0] + "\t" + str(elem[1]) + "\t" + str(elem[2]) + "\t" + str(elem[3]) + "\t" + str(elem[4]))
        """
    
        saved_data = data
        return data

    elif saved_data is not None:
        return saved_data
    


def perceptron(rho, nit, forced):

    global saved_data, saved_ws

    data = saved_data

    if saved_data is None:
        print("ERROR - Perceptron: El dataset es null!")
        return

    if forced or saved_ws is None:
    
        n = len(data)       # Número de elementos en el dataset de entrenamiento
        
        w0 = random.random()
        w1 = random.random()
        w2 = random.random()
        #w3 = math.sqrt(1-(math.pow(w2, 2)))
        w3 = random.random()
        
        w = numpy.transpose(numpy.asmatrix([w1, w2, w3, w0]))

        print("\nW inicializados aleatoriamente: " + str([w1, w2, w3, w0]))


        x = []
        c_1 = []    # Característica 1 (Hue)
        c_2 = []    # Característica 2 (Saturación)
        c_3 = []    # Característica 3 (Energía)
        for i in range(len(data)):
            c_1.append(data[i][1])
            c_2.append(data[i][2])
            c_3.append(data[i][3])
        x.append(c_1)
        x.append(c_2)
        x.append(c_3)

        ones_line = numpy.ones((1, n), dtype=int)   # Linea de 1s
        x.append(ones_line[0])

        best_ic = sys.maxsize   # Mejor número de clasificaciones incorrectas
        best_w = w              # Ws correspondientes al mejor número de clasificaciones incorrectas
        contador = 0

        for t in range(nit):
            sum = numpy.zeros((4, 1), dtype=int)
            ic = 0

            contador += 1
            if contador % 25 == 0:
                print("Entrenando... - {:.2f}%  ".format(t/nit * 100), end="\r") # Visualiza el progreso en %

            for k in range(n):
                xi = []
                for i in range(len(x)):
                    xi.append([x[i][k]])
                xi = numpy.asarray(xi)
                
                #print("DEBUG - " + str(numpy.dot(numpy.transpose(w), xi)))

                if numpy.dot(numpy.transpose(w), xi) < 0 and data[k][4] == 1:
                    sum = numpy.add(sum, (rho* xi))
                    ic += 1
                elif numpy.dot(numpy.transpose(w), xi) > 0 and data[k][4] == 2:
                    sum = numpy.subtract(sum, (rho * xi))
                    ic += 1
            
            # Variante del bolsillo: no nos quedamos con la última iteración, si no con la que ha dado los mejores resultados
            if ic < best_ic:
                #print("Encontrado IC mejor! | ic = {:d}, best_ic previo = {:d}".format(ic, best_ic))
                best_w = w
                best_ic = ic
            
            w = numpy.add(w, sum)

            #print("W para iteración {:d}:\t".format(t), end="")
            #print(str(numpy.transpose(w)) + " | ic=" + str(ic) + " | best_ic=" + str(best_ic))
            
            # Si la iteración consigue clasificar correctamente todos los elementos, ya no hace falta seguir entrenando
            if ic == 0:
                break

        print("Perceptron - 100% | Acabado!")
        
        print("W ha quedado: " + str(numpy.transpose(best_w)) + " | Mejor número de incorrectos: " + str(best_ic))

        saved_ws = best_w

        # Paso de array/matriz de numpy a lista normal
        w_s = saved_ws.tolist()
        w_x = []
        for elem in w_s:
            w_x.append(elem[0])
        print(w_x)

        figure = matplotlib.pyplot.figure()

        ax = matplotlib.pyplot.axes(projection="3d")

        # crea valor pels eixos x,y
        x = y = numpy.arange(0.0, 1.0, 0.1)
        [xx,yy] = numpy.meshgrid(x, y)
        # calcula els valors de z
        z = (-w_x[0]*xx - w_x[1]*yy - w_x[3])/w_x[2]
        # pinta el pla de separació
        ax.plot_surface(xx, yy, z, alpha=0.7)

        # Separa los datos de entreno por clase
        data_1 = []
        data_2 = []

        for i in range(len(data)):
            temp = []
            for j in range(1,4):
                temp.append(data[i][j])
            if i < 491:
                data_1.append(temp)
            else:
                data_2.append(temp)

        matx_1 = numpy.asmatrix(data_1)
        matx_2 = numpy.asmatrix(data_2)

        ax.scatter(matx_1[:, 0], matx_1[:, 1], matx_1[:, 2], c="blue")
        ax.scatter(matx_2[:, 0], matx_2[:, 1], matx_2[:, 2], c="red")

        print("Enseñando hyperplano 3D - Clase 1 en Azul, Clase 2 en Rojo")
        figure.show()

    elif saved_ws is not None:
        print("Perceptron ya entrenado, usando valores de W: " + str(numpy.transpose(saved_ws)))

    return saved_ws

def clasificaImagen(coordenadas, index_img):

    global saved_ws

    try:
        img = imgs[index_img-1]
    except:
        print("ERROR - Index de la imagen incorrecto ({:d})".format(index_img))
        return None
    
    # Repartición de la imagen en segmentos

    ws = saved_ws
    segmentos = []
    
    coords = coordenadas
    coords.append(0.0)                  # Añadimos el borde izquierdo
    coords.append(float(img.shape[1]))  # Añadimos el borde derecho
    coords = sorted(coords)
    print("Coordenadas X de los bordes de los segmentos: " + str(coords))
    height = img.shape[0]

    for i in range(len(coords)-1):
        segmentos.append(img[0:height, int(coords[i]):int(coords[i+1])])
        #print("DEBUG - Segmento {:d}\tDe {:f} a {:f}".format(i, coords[i], coords[i+1]))
    
    """
    for elem in segmentos:
        figure = matplotlib.pyplot.figure()
        skimage.io.imshow(elem)
        figure.show()
    """
    # Genera fragmentos de 30x30 px para cada segmento

    patches = []

    for segmento in segmentos:
        width = segmento.shape[1]
        height = segmento.shape[0]
        patches_segm = []

        n_horiz = math.floor(width/30)
        n_vert = math.floor(height/30)

        print("Tamaño segmento | W: {:d}  H: {:d} | patches h: {:d} v: {:d} | Total patches esperadas: {:d}".format(width, height, n_horiz, n_vert, (n_horiz*n_vert)))

        for i in range(n_vert):
            for j in range(n_horiz):
                #print("\tCortando fragmento {:d},{:d}\t| De {:d},{:d} a {:d},{:d}\t| Tamaño esperado: {:d}x{:d}".format(i, j, 30*i, 30*(i+1), 30*j, 30*(j+1), (30*(i+1-i)), (30*(j+1-j))))
                patches_segm.append(segmento[(30*i):(30*(i+1)), (30*j):(30*(j+1)), :])  # Los : al final son para indicar que coja todos los canales, no sé si hace falta pero no lo vuelvo a tocar

                
        print("Fragmentación acabada, n. de fragmentos: {:d}".format(len(patches_segm)))
        
        """
        print("Tamaños: ", end="")
        for p in range(len(patches_segm)):
            print("{:d}x{:d}, ".format(patches_segm[p].shape[0], patches_segm[p].shape[1]), end="")
            #skimage.io.imsave(("test/patch" + str(contador) + "_" + str(p+1) + "_of_" + str(len(patches_segm)) + ".jpg"), patches_segm[p])
        print("")
        """

        patches.append(patches_segm)

    # Clasificación de los segmentos de la imagen

    resultados = []
    test_ws = ws[0:3]

    print("Clasificando", end="")

    for i in range(len(patches)):   # Para cada segmento
        resultados.append({
            "1" : 0,
            "2" : 0
        })
        print(".", end="")

        #print("\tDEBUG - len patches[{:d}]: {:d}".format(i, len(patches[i])))        

        for j in range(len(patches[i])):    # Para cada fragmento de cada segmento
            # Test con perceptron y clasificación

            patch_hsv = skimage.color.rgb2hsv(patches[i][j])
            hue = numpy.average(patch_hsv[0])
            saturation = numpy.average(patch_hsv[1])
        
            glcm = skimage.feature.greycomatrix(skimage.img_as_ubyte(skimage.color.rgb2gray(patches[i][j])), distances=[1], angles=[0], levels=256, normed=True)
            energy = skimage.feature.greycoprops(glcm, prop='energy')
            
            chars = [hue, saturation, energy[0][0]]

            #print("DEBUG - w:     " + str(ws))
            #print("DEBUG - chars: " + str(chars))

            resultado = numpy.dot(numpy.transpose(test_ws), numpy.transpose(chars)) + ws[3]

            #print("Resultado para patch {:d} {:d}: ".format(i, j) + str(resultado), end="")

            if resultado[0][0] > 0:
                resultados[i]["1"] += 1
            else:
                resultados[i]["2"] += 1

    print(" Acabado!")            

    # Cálculo de los porcentajes y verdicto final para cada segmento
    verdictos = []
    print("\n***Recuento final***")
    for i in range(len(resultados)):
        print("Valores para segmento " + str(i) + ": " + str(resultados[i]))
        total = resultados[i]["1"] + resultados[i]["2"]
        if total != 0:
            p_1 = (resultados[i]["1"] / total) * 100
            p_2 = (resultados[i]["2"] / total) * 100
        else:
            p_1 = p_2 = 0.0
        print("\tPorcentajes: {:.2f}% puerta\t{:.2f}% pared".format(p_1, p_2))

        if p_1 > 70.0:
            print("\tVerdicto: PUERTA")
            verdictos.append(1)
        elif p_2 > 60.0:
            print("\tVerdicto: PARED")
            verdictos.append(2)
        else:
            print("\tVerdicto: INDETERMINADO")
            verdictos.append(0)

    # Pintar la imagen
    img_2 = imgs[index_img-1]
    fig = matplotlib.pyplot.figure()
    skimage.io.imshow(img_2)

    for i in range(len(coords)-1):

        if verdictos[i] == 0:   # Indeterminado
            matplotlib.pyplot.gca().add_patch(matplotlib.patches.Rectangle((coords[i], 0), coords[i+1]-coords[i], img_2.shape[0], linewidth=0, edgecolor="white", facecolor="white"))
        elif verdictos[i] == 2: # Pared
            matplotlib.pyplot.gca().add_patch(matplotlib.patches.Rectangle((coords[i], 0), coords[i+1]-coords[i], img_2.shape[0], linewidth=0, edgecolor="black", facecolor="black"))
    
    fig.show()
    
    return verdictos

        
if __name__ == "__main__":
    
    print("*** Práctica 3 - Percepción ***\nAsignatura de Percepción y Control para Sistemas Empotrados\nIzar Castorina - Curso 2020/2021")
    print("Cargando imagenes", end="")
    for i in range(1, 6):
        # Carga las 5 imagenes en la lista imgs
        imgs.append(skimage.io.imread("image"+str(i)+".JPG"))
        print(".", end="")
    print(" Carga completada.")

    coordLineas = [None, None, None, None, None]

    forced = False
    
    while True:
        print("\n***************\nOpciones")
        print("\t1-5\tEjecuta el script para la imagen correspondiente")
        print("\t6\tEjecuta el script para todas las imagenes de forma secuencial")
        print("\t7\tModifica el valor de la rho del Perceptron")
        print("\t8\tModifica el número de iteraciones del Perceptron")
        print("\t9\tSal del programa")

        
        print("\n\t0\tHabilita/Deshabilita el reentrenamiento forzado del Perceptrón")
        print("\t\tEstado actual: ", end="")
        print("Activo", end="") if forced else print("Desactivado", end="")
        print("\tEstado Perceptron: ", end="")
        print("No entrenado") if saved_ws is None else print("Entrenado")
                
        print("\n\ts\tHabilita/Deshabilita el stretching en la visualización de los acumuladores de Hough")
        print("\t\tEstado actual: ", end="")
        print("Activo") if enable_stretched else print("Desactivado")
        
        print("\tr\tHabilita/Deshabilita la visualización solamente de los resultados finales")
        print("\t\tEstado actual: ", end="")
        print("Activo") if only_show_result else print("Desactivado")

        print("\n\tValores actuales para el Perceptron | Rho = {:f} | N.iteraciones = {:d}".format(perc_rho, perc_nit))
        
        print("")
                
        sel = input("Tu selección: ")

        print("")
        
        if sel == "1" or sel == "2" or sel == "3" or sel == "4" or sel == "5":
            coordLineas[int(sel)-1] = trazaLineas(int(sel))
            obtenerCaracteristicas(forced)
            perceptron(perc_rho, perc_nit, forced)
            clasificaImagen(coordLineas[int(sel)-1], int(sel))
        elif sel == "6":
            for i in range(1, 6):
                coordLineas[i-1] = trazaLineas(i)
                obtenerCaracteristicas(forced)
                perceptron(perc_rho, perc_nit, forced)
                clasificaImagen(coordLineas[i-1], i)

        elif sel == "7":
            new_rho = input("Inserta un valor de Rho para el Perceptron: ")
            try:
                new_rho_f = float(new_rho)
                if new_rho_f <= 0.0:
                    print("Error - Rho tiene que ser mayor que 0.")
                else:
                    perc_rho = new_rho_f
                    reent = input("** Los parámetros para el Perceptron han sido modificados. Reentrenar? (s/n) ")
                    if reent == "s" or reent == "S":
                        obtenerCaracteristicas(False)
                        perceptron(perc_rho, perc_nit, True)
            except:
                print("Error - Valor no reconocido.")

        elif sel == "8":
            new_nit = input("Inserta el nuevo número de iteraciones del Perceptron: ")
            try:
                new_nit_int = int(new_nit)
                if new_nit_int <= 0:
                    print("Error - El número de iteraciones tiene que ser mayor que 0.")
                else:
                    perc_nit = new_nit_int
                    reent = input("** Los parámetros para el Perceptron han sido modificados. Reentrenar? (s/n) ")
                    if reent == "s" or reent == "S":
                        obtenerCaracteristicas(False)
                        perceptron(perc_rho, perc_nit, True)

            except:
                print("Error - Valor no reconocido.")

        elif sel == "0":
            forced = not forced

        elif sel == "9":
            print("Saliendo...")
            exit(0)
            
        elif sel == "s" or sel == "S":
            enable_stretched = not enable_stretched
        elif sel == "r" or sel == "R":
            only_show_result = not only_show_result
            
        else:
            print("Comando no reconocido.")