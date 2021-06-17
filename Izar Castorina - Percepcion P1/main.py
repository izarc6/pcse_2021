import numpy
import skimage.io
import skimage.transform
import skimage.color
import matplotlib.pyplot
import skimage.exposure
import skimage.filters
import skimage.feature
from skimage.morphology import disk

import warnings


def parte1():
    print("Ejecución parte 1... ", end="")
    # Cargo la imagen
    img_1 = skimage.io.imread("practica1_1.jpg")

    # Resize a 1:2, 1:4 y 1:16
    new_size = (int(img_1.shape[0] * 0.5), int(img_1.shape[1] * 0.5))
    img_1_2 = skimage.transform.resize(img_1, new_size)
    new_size = (int(img_1.shape[0] * 0.25), int(img_1.shape[1] * 0.25))
    img_1_4 = skimage.transform.resize(img_1, new_size)
    new_size = (int(img_1.shape[0] * 0.0625), int(img_1.shape[1] * 0.0625))
    img_1_16 = skimage.transform.resize(img_1, new_size)

    # Guardo las imagenes escaladas
    skimage.io.imsave("escalat1_02.jpg", img_1_2)
    skimage.io.imsave("escalat1_04.jpg", img_1_4)
    skimage.io.imsave("escalat1_16.jpg", img_1_16)

    # Visualiza las imagenes en el mismo plot
    figure = matplotlib.pyplot.figure()
    figure.add_subplot(2, 2, 1)
    skimage.io.imshow(img_1)
    figure.add_subplot(2, 2, 2)
    skimage.io.imshow(img_1_2)
    figure.add_subplot(2, 2, 3)
    skimage.io.imshow(img_1_4)
    figure.add_subplot(2, 2, 4)
    skimage.io.imshow(img_1_16)
    figure.show()
    
    # Pasaje a escala de grises y guardado de la imágen
    img_1_gray = skimage.color.rgb2gray(img_1)
    skimage.io.imsave("gray_1.jpg", img_1_gray)
    
    print("Completada.")

def parte2():
    print("Ejecución parte 2... ", end="")
    # Cargo la segunda imagen
    img_2 = skimage.io.imread("practica1_2.jpg")

    # Visualización del histograma
    hist_2, hist_2_centers = skimage.exposure.histogram(img_2, nbins = 2)

    hist = matplotlib.pyplot.figure()
    matplotlib.pyplot.bar(hist_2_centers, hist_2)
    matplotlib.pyplot.title("Histograma original")
    hist.show()

    # Ejecución del stretching del histograma
    img_2_resc = skimage.exposure.rescale_intensity(img_2)

    # Visualización del histograma actualziado
    hist_2_resc, hist_2_resc_centers = skimage.exposure.histogram(img_2_resc, nbins = 2)
    hist_resc = matplotlib.pyplot.figure()
    matplotlib.pyplot.bar(hist_2_resc_centers, hist_2_resc)
    matplotlib.pyplot.title("Histograma después de stretching")
    hist_resc.show()

    # Ecualización del histograma
    img_2_ubyte = skimage.img_as_ubyte(img_2)
    img_2_eq = skimage.exposure.equalize_hist(img_2_ubyte)

    # Distribución de los niveles de la imagen ecualizada
    dist, dist_centers = skimage.exposure.cumulative_distribution(img_2_eq)
    dist_plot_eq = matplotlib.pyplot.figure()
    matplotlib.pyplot.bar(dist_centers, dist)
    matplotlib.pyplot.title("Distribución cumulativa (ecualizada)")
    dist_plot_eq.show()

    # Ecualización por CLAHE
    img_2_clahe = skimage.exposure.equalize_adapthist(img_2_ubyte)

    # Distribución de los niveles de la imagen ecualizada por CLAHE
    dist_clahe, dist_centers_clahe = skimage.exposure.cumulative_distribution(img_2_clahe)
    dist_plot_clahe = matplotlib.pyplot.figure()
    matplotlib.pyplot.bar(dist_centers_clahe, dist_clahe)
    matplotlib.pyplot.title("Distribución cumulativa (CLAHE)")
    dist_plot_clahe.show()

    # Visualización de las 4 imagenes
    figure = matplotlib.pyplot.figure()
    ax1 = figure.add_subplot(2, 2, 1)
    ax1.title.set_text("Imagen original")
    skimage.io.imshow(img_2)
    ax2 = figure.add_subplot(2, 2, 2)
    ax2.title.set_text("Con stretching del histograma")
    skimage.io.imshow(img_2_resc)
    ax3 = figure.add_subplot(2, 2, 3)
    ax3.title.set_text("Ecualizada")
    skimage.io.imshow(img_2_eq)
    ax4 = figure.add_subplot(2, 2, 4)
    ax4.title.set_text("Ecualizada con CLAHE")
    skimage.io.imshow(img_2_clahe)
    figure.show()

    # Visualización de los cuatro histogramas
    figure, ax = matplotlib.pyplot.subplots(nrows=2, ncols=2)
    ax[0, 0].bar(hist_2_centers, hist_2)
    ax[0, 0].title.set_text("Histograma original")

    ax[0, 1].bar(hist_2_resc_centers, hist_2_resc)
    ax[0, 1].title.set_text("Histograma con stretching")

    a, b = skimage.exposure.histogram(img_2_eq*255)
    ax[1, 0].bar(b, a)
    ax[1, 0].title.set_text("Histograma ecualizado")

    c, d = skimage.exposure.histogram(img_2_clahe*255)
    ax[1, 1].bar(d, c)
    ax[1, 1].title.set_text("Histograma (CLAHE)")

    figure.tight_layout()
    matplotlib.pyplot.show()

    # Eliminación de ruido de Gauss
    img_gauss = skimage.io.imread("practica1_3_gauss.jpg")

    # Filtro de media - disco de radio = 5
    img_mean = skimage.filters.rank.mean(img_gauss, disk(5))
    # Filtro Gaussiano
    img_gaussFilter = skimage.filters.gaussian(img_gauss)
    # Filtro de mediana
    img_median = skimage.filters.rank.median(img_gauss, disk(5))
    # Visualización
    figure = matplotlib.pyplot.figure()
    ax1 = figure.add_subplot(2, 2, 1)
    ax1.title.set_text("Imagen original (Gauss)")
    skimage.io.imshow(img_gauss)
    ax2 = figure.add_subplot(2, 2, 2)
    ax2.title.set_text("Filtro de media")
    skimage.io.imshow(img_mean)
    ax3 = figure.add_subplot(2, 2, 3)
    ax3.title.set_text("Filtro Gaussiano")
    skimage.io.imshow(img_gaussFilter)
    ax4 = figure.add_subplot(2, 2, 4)
    ax4.title.set_text("Filtro de mediana")
    skimage.io.imshow(img_median)
    figure.tight_layout()
    figure.show()

    # Eliminación de ruido sal y pimienta
    img_sp = skimage.io.imread("practica1_3_saltpepper.jpg")

    # Filtro de media - disco de radio = 5
    img_mean = skimage.filters.rank.mean(img_gauss, disk(5))
    # Filtro Gaussiano
    img_gaussFilter = skimage.filters.gaussian(img_sp)
    # Filtro de mediana
    img_median = skimage.filters.rank.median(img_sp, disk(5))
    # Visualización
    figure = matplotlib.pyplot.figure()
    ax1 = figure.add_subplot(2, 2, 1)
    ax1.title.set_text("Imagen original (S&P)")
    skimage.io.imshow(img_sp)
    ax2 = figure.add_subplot(2, 2, 2)
    ax2.title.set_text("Filtro de media")
    skimage.io.imshow(img_mean)
    ax3 = figure.add_subplot(2, 2, 3)
    ax3.title.set_text("Filtro Gaussiano")
    skimage.io.imshow(img_gaussFilter)
    ax4 = figure.add_subplot(2, 2, 4)
    ax4.title.set_text("Filtro de mediana")
    skimage.io.imshow(img_median)
    figure.tight_layout()
    figure.show()

    print("Completada.")

def parte3():
    print("Ejecución parte 3... ", end="")
    # Binarización de imagenes
    imgs = [skimage.io.imread("practica1_4.jpg"), skimage.io.imread("practica1_5.jpg"),
            skimage.io.imread("practica1_6.jpg"),skimage.io.imread("practica1_7.jpg")]
    imgs_thresh = [[], [], []]
    thresholds = [0.2, 0.5, 0.7]

    # Binarizaciones con thresholds definidas
    for i in range(4):
        for j in range(3):
            temp = imgs[i] < (thresholds[j]*255) # Dato que las intensidades van de 0 a 255
            imgs_thresh[j].append(temp)

    # Binarización con método de Otsu
    imgs_otsu = []
    otsu = []
    for i in range(4):
        otsu.append(skimage.filters.threshold_otsu(imgs[i]))
        imgs_otsu.append(imgs[i] <= otsu[i])

    # Visualiza las imagenes
    for i in range(4):
        figure = matplotlib.pyplot.figure()
        ax1 = figure.add_subplot(2, 2, 1)
        ax1.title.set_text("Imagen original " + str(i + 4))
        skimage.io.imshow(imgs[i].astype(numpy.uint8), cmap='gray')
        for j in range(3):
            ax = figure.add_subplot(2, 2, 2+j)
            ax.title.set_text("Binarización con " + str(thresholds[j]))
            skimage.io.imshow(imgs_thresh[j][i].astype(numpy.uint8), cmap='gray')
        figure.tight_layout()
        figure.show()

    figure = matplotlib.pyplot.figure()
    for j in range(4):
        ax = figure.add_subplot(2, 2, 1 + j)
        ax.title.set_text("Imagen " + str(j+4) + " | Otsu = " + str("{:.4f}".format(otsu[j]/255))) # /255 para tener un valor normalizado, redondeo a 4 cifras
        skimage.io.imshow(imgs_otsu[j].astype(numpy.uint8), cmap='gray')
    figure.tight_layout()
    figure.show()
    print("Completada.")

def parte4():
    print("Ejecución parte 4... ", end="")
    # Extracción de contornos
    imgs = [skimage.io.imread("practica1_4.jpg"), skimage.io.imread("practica1_5.jpg"),
            skimage.io.imread("practica1_6.jpg"), skimage.io.imread("practica1_7.jpg")]
    sobel, roberts, prewitt, canny = ([] for i in range(4))

    for i in range(4):
        sobel.append(skimage.filters.sobel(imgs[i]))
        roberts.append(skimage.filters.roberts(imgs[i]))
        prewitt.append(skimage.filters.prewitt(imgs[i]))
        canny.append(skimage.feature.canny(imgs[i]))

    for i in range(4):
        figure = matplotlib.pyplot.figure()
        ax1 = figure.add_subplot(2, 2, 1)
        ax1.title.set_text("Imagen " + str(i+4) + " | Sobel")
        skimage.io.imshow(sobel[i])
        ax2 = figure.add_subplot(2, 2, 2)
        ax2.title.set_text("Imagen " + str(i + 4) + " | Roberts")
        skimage.io.imshow(roberts[i])
        ax3 = figure.add_subplot(2, 2, 3)
        ax3.title.set_text("Imagen " + str(i + 4) + " | Prewitt")
        skimage.io.imshow(prewitt[i])
        ax4 = figure.add_subplot(2, 2, 4)
        ax4.title.set_text("Imagen " + str(i + 4) + " | Canny")
        skimage.io.imshow(canny[i].astype(numpy.uint8), cmap='gray')
        figure.tight_layout()
        figure.show()

    print("Completada.")


if __name__ == '__main__':
    warnings.filterwarnings("ignore") # Elimina los warnings
    matplotlib.pyplot.rcParams['figure.figsize'] = (12.8, 10.24) # Hace los plots más grandes

    print("***  Práctica 1 - Percepción  ***\n*** Izar Castorina, 2020-2021 ***\n")
    while True:
        print("\t1. Parte 1 - Escalado y visualización de una imagen")
        print("\t2. Parte 2 - Mejora de imágenes y eliminación del ruido")
        print("\t3. Parte 3 - Binarización de imágenes")
        print("\t4. Parte 4 - Extracción de contornos")
        print("\t5. Todas las partes")
        print("\t9. Salir")
        sel = input("Qué quieres ejecutar? ")
        if sel == "1":
            parte1()
        elif sel == "2":
            parte2()
        elif sel == "3":
            parte3()
        elif sel == "4":
            parte4()
        elif sel == "5":
            parte1()
            parte2()
            parte3()
            parte4()
        elif sel == "9":
            print("Saliendo...")
            break
        else:
            print("Comando no reconocido.")
        print("")