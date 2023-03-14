import matplotlib.pyplot as plt
import matplotlib.colors as clr

import cv2
import numpy as np

import scipy.fftpack as fft

import math

from ntpath import join

##TODO
# Remover o ciclo do join channels
ycbcr_matrix = np.array([
    [65.481, 128.553, 24.966],
    [-37.797, -74.203, 112.0],
    [112.0, -93.786, -18.214]])

#3.1

def read_image(img_path):
    return plt.imread(img_path)

#3.2

def colormap_function(colormap_name, color1, color2):
    return clr.LinearSegmentedColormap.from_list(colormap_name, [color1, color2], 256)

#3.3

def draw_plot(image, colormap):
    plt.figure()
    plt.imshow(image, colormap)
    plt.title("Imagem original")
    plt.show()


#3.4

def rgb_components(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    return R, G, B

def rgb_components_reverse(R, G, B):
    gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1])
    
    matrix_inverted = np.zeros((len(R), len(R[0]), 3), dtype=np.uint8)
    matrix_inverted[:, :, 0] = R
    matrix_inverted[:, :, 1] = G
    matrix_inverted[:, :, 2] = B
    
    fig = plt.figure()
    plt.title("rgb components reverse")
    plt.imshow(matrix_inverted, cmap = gray_colormap)

    return matrix_inverted

#3.5

def show_rgb(c_R, c_G, c_B):
    cm_red = colormap_function("Red", (0, 0, 0), (1, 0, 0))
    cm_green = colormap_function("Green", (0, 0, 0), (0, 1, 0))
    cm_blue = colormap_function("Blue", (0, 0, 0), (0, 0, 1))
    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.title("Channel R")
    plt.imshow(c_R, cm_red)
    fig.add_subplot(1, 3, 2)
    plt.title("Channel G")
    plt.imshow(c_G, cm_green)
    fig.add_subplot(1, 3, 3)
    plt.title("Channel B")
    plt.imshow(c_B, cm_blue)
    plt.subplots_adjust(wspace=0.5)
    plt.show()

#4

def padding(imgFile):
    img_bgr = cv2.imread(imgFile)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    #Linhas
    nl = img.shape[0]
    #colunas
    nc = img.shape[1]

    padding_linhas = 32- (nl % 32)
    padding_colunas = 32- (nc % 32)

    ll = img[nl - 1, :][np.newaxis, :]
    repl = ll.repeat(padding_linhas, axis=0)
    imgp = np.vstack([img, repl])

    lc = imgp[:, nc-1][:, np.newaxis]
    repc = lc.repeat(padding_colunas, axis=1)
    imgp = np.hstack([imgp, repc])
    
    plt.figure()
    plt.imshow(imgp,None)
    plt.title("Imagem com padding")
    plt.show()
    
    return imgp,nl,nc

def reverse_padding(image_array, nl_original, nc_original):
    # Extract the unpadded image
    unpadded_image = image_array[:nl_original, :nc_original ]
    
    plt.figure()
    plt.imshow(unpadded_image)
    plt.title("Imagem sem padding")
    plt.show()
    
    return unpadded_image
    
#5

def RGB_toYCbCr(R, G, B, ImageFile):
    
    ycbr = np.dot(ImageFile,ycbcr_matrix.T)
    
    R_aux= ycbr[:,:,0] + 128
    G_aux= ycbr[:,:,1] + 128
    B_aux= ycbr[:,:,2] + 128
    
    
    Y = (0.299 * R) + (0.587 * G) + (0.114 * B)
    Cb = (-0.168736 * R) + (-0.331264 * G) + (0.5 * B) + 128
    Cr = (0.5 * R) + (-0.418688 * G) + (-0.081312 * B) + 128

    return Y, Cb, Cr, ycbr

def YCbCr_to_RGB(Y, Cb, Cr):
    matrix = np.array([[0.299,   0.587,    0.114],
                       [-0.168736, -0.331264,     0.5],
                       [0.5, -0.418688, -0.081312]])
    T_matrix = np.linalg.inv(matrix)

    R = np.round(T_matrix[0][0] * Y + T_matrix[0][1] * (Cb - 128) + T_matrix[0][2] * (Cr - 128))
    R[R < 0] = 0
    R[R > 255] = 255

    G = np.round(T_matrix[1][0] * Y + T_matrix[1][1] *(Cb - 128) + T_matrix[1][2] * (Cr - 128))
    G[G < 0] = 0
    G[G > 255] = 255

    B = np.round(T_matrix[2][0] * Y + T_matrix[2][1] * (Cb - 128) + T_matrix[2][2] * (Cr - 128))
    B[B < 0] = 0
    B[B > 255] = 255

    return np.uint8(R), np.uint8(G), np.uint8(B)  

def showCanals(ycbcr_image):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1])

    # Plota o canal Y com colormap gray
    ax1.imshow(ycbcr_image[:, :, 0], cmap=gray_colormap)
    ax1.set_title("Canal Y")

    # Plota o canal Cb com colormap jet
    ax2.imshow(ycbcr_image[:, :, 1], cmap=gray_colormap)
    ax2.set_title("Canal Cb")

    # Plota o canal Cr com colormap jet
    ax3.imshow(ycbcr_image[:, :, 2], cmap=gray_colormap)
    ax3.set_title("Canal Cr")
    
    plt.title("Mostra os canais Y, Cb, Cr da imagem")
    plt.show()
    
#6
    
def downsampling(Y, Cb, Cr, type):
    
    gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1])
    
    if type == 0:
        scaleX = 0.5
        scaleY = 0.5
        Cb_d = cv2.resize(Cb, None, fx=scaleX, fy=scaleY,interpolation=cv2.INTER_LINEAR)
        Cr_d = cv2.resize(Cr, None, fx=scaleX, fy=scaleY,interpolation=cv2.INTER_LINEAR)
        
        fig = plt.figure()
    
        fig.add_subplot(3, 2, 3)
        plt.title("Cb - Downsampling 4:2:0")
        plt.imshow(Cb_d, cmap=gray_colormap)
    
        fig.add_subplot(3, 2, 4)
        plt.title( "Cr - Downsampling 4:2:0")
        plt.imshow(Cr_d, cmap=gray_colormap)
    
        plt.subplots_adjust(hspace=0.5)
        
        plt.show()

    else:
        scaleX = 0.5
        scaleY = 1
        Cb_d = cv2.resize(Cb, None, fx=scaleX, fy=scaleY,interpolation=cv2.INTER_LINEAR)
        Cr_d = cv2.resize(Cr, None, fx=scaleX, fy=scaleY,interpolation=cv2.INTER_LINEAR)
        
        fig = plt.figure()
    
        fig.add_subplot(3, 2, 3)
        plt.title("Cb - Downsampling 4:2:2")
        plt.imshow(Cb_d, cmap=gray_colormap)

        fig.add_subplot(3, 2, 4)
        plt.title( "Cr - Downsampling 4:2:2")
        plt.imshow(Cr_d, cmap=gray_colormap)

        plt.subplots_adjust(hspace=0.5)

        plt.show()

    return Y, Cb_d, Cr_d

def upsampling(Y_d, Cb_d, Cr_d, type):
    
    gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1])
    
    if type == 0:

        scaleX = 0.5
        scaleY = 0.5
        stepX = int(1//scaleX)
        stepY = int(1//scaleY)

        Cb_u = cv2.resize(Cb_d, None, fx=stepX, fy=stepY,interpolation=cv2.INTER_LINEAR)
        Cr_u = cv2.resize(Cr_d, None, fx=stepX, fy=stepY,interpolation=cv2.INTER_LINEAR)
        
        fig = plt.figure()
    
        fig.add_subplot(3, 2, 3)
        plt.title("Cb - Upsampling 4:2:0")
        plt.imshow(Cb_u, cmap=gray_colormap)

        fig.add_subplot(3, 2, 4)
        plt.title( "Cr - Upsampling 4:2:0")
        plt.imshow(Cr_u, cmap=gray_colormap)

        plt.subplots_adjust(hspace=0.5)

        plt.show()
    else:
        scaleX = 0.5
        scaleY = 1
        stepX = int(1//scaleX)
        stepY = int(1//scaleY)
        
        Cb_u = cv2.resize(Cb_d, None, fx=stepX, fy=stepY,interpolation=cv2.INTER_LINEAR)
        Cr_u = cv2.resize(Cr_d, None, fx=stepX, fy=stepY,interpolation=cv2.INTER_LINEAR)
        
        fig = plt.figure()
    
        fig.add_subplot(3, 2, 3)
        plt.title("Cb - Upsampling 4:2:2")
        plt.imshow(Cb_u, cmap=gray_colormap)

        fig.add_subplot(3, 2, 4)
        plt.title( "Cr - Upsampling 4:2:2")
        plt.imshow(Cr_u, cmap=gray_colormap)

        plt.subplots_adjust(hspace=0.5)

        plt.show()

    return Y_d, Cb_u, Cr_u
    
def dct(channel, chanel_name, blocks, channel_0, type):
    
    gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1]) 
    
    if type == 1:       
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        channel_dct = fft.dct(fft.dct(channel, norm="ortho").T, norm="ortho").T

        ax1.imshow(channel_dct, cmap=gray_colormap)
        ax1.set_title("DCT do canal " + chanel_name + " completo")
        
        channel_dct_log = np.log(np.abs(channel_dct))
        
        ax2.imshow(channel_dct_log, cmap=gray_colormap)
        ax2.set_title("DCT do canal " + chanel_name + " completo com log")
        
        channel_dct = np.zeros(channel.shape)
        channel_dct_log = np.zeros(channel.shape)
        
        for i in range(0, len(channel), blocks):
            for j in range(0, len(channel[0]), channel_0):
                channel_dct[i:i+blocks, j:j+channel_0] = fft.dct(fft.dct(channel[i:i+blocks, j:j+channel_0], norm="ortho").T, norm="ortho").T
                channel_aux = fft.dct(fft.dct(channel[i:i+blocks, j:j+channel_0], norm="ortho").T, norm="ortho").T
                channel_dct_log[i:i+blocks, j:j +channel_0] = np.log(np.abs(channel_aux) + 0.0001)
                
        ax3.imshow(channel_dct_log, cmap=gray_colormap)
        ax3.set_title("DCT do canal " + chanel_name + " em blocos com log")
        
        plt.show()
        
    else:
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        channel_dct = fft.dct(fft.dct(channel, norm="ortho").T, norm="ortho").T
    
        ax1.imshow(channel_dct, cmap=gray_colormap)
        ax1.set_title("DCT do canal " + chanel_name + " completo")
        
        channel_dct_log = np.log(np.abs(channel_dct))
        
        ax2.imshow(channel_dct_log, cmap=gray_colormap)
        ax2.set_title("DCT do canal " + chanel_name + " completo com log")
        
        plt.show()

    return channel_dct, channel_dct_log

def dct_inverse(channel_dct,  blocks, channel_0):
    
    gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1]) 
    
    channel_d = np.zeros(channel_dct.shape)
    channel_d_block = np.zeros(channel_dct.shape)

    for i in range(0, len(channel_dct), blocks):
        for j in range(0, len(channel_dct[0]), channel_0):
            channel_da = channel_dct[i:i+blocks, j:j+channel_0]
            channel_d[i:i+blocks, j:j+channel_0] = fft.idct(fft.idct(channel_da, norm="ortho").T, norm="ortho").T

    plt.imshow(channel_d, cmap=gray_colormap)
    plt.title("DCT inverso canal completo")
    
    plt.imshow(channel_d, cmap=gray_colormap)
    plt.title("DCT inverso canal em blocos")
    
    plt.show()

    return channel_d, channel_d_block

#8

def quality_factor(Q, qf):
    if(qf >= 50):
        sf = (100 - qf) / 50
    else:
        sf = 50/qf

    if(sf != 0):
        Qs = np.round(Q * sf)
    else:
        Qs = np.ones(Q.shape, dtype=np.uint8)

    Qs[Qs > 255] = 255
    Qs[Qs < 1] = 1
    
    Qs = Qs.astype(np.uint8)

    return Qs

def quantized_dct_coefficients_8x8(Y_dct, Cb_dct, Cr_dct):
    qualities = np.array([10, 25, 50, 75, 100])
    
    Q_Y = np.array([[16,  11,  10,  16,  24,  40,  51,  61],
                    [12,  12,  14,  19,  26,  58,  60,  55],
                    [14,  13,  16,  24,  40,  57,  69,  56],
                    [14,  17,  22,  29,  51,  87,  80,  62],
                    [18,  22,  37,  56,  68, 109, 103,  77],
                    [24,  35,  55,  64,  81, 104, 113,  92],
                    [49,  64,  78,  87, 103, 121, 120, 101],
                    [72,  92,  95,  98, 112, 100, 103,  99]])
    
    Q_Y_with_tile = np.tile(Q_Y, (int(len(Y_dct)/8), int(len(Y_dct[0])/8)))

    Q_CbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])
    
    Q_CbCr_with_tile = np.tile(Q_CbCr, (int(len(Cb_dct)/8), int(len(Cb_dct[0])/8)))
    
    for i in qualities:
        quality_factor(Q_Y_with_tile, i)
        quality_factor(Q_CbCr_with_tile, i)

        quantized_Y_dct = np.round(Y_dct / Q_Y_with_tile).astype(int)
        quantized_Cb_dct = np.round(Cb_dct / Q_CbCr_with_tile).astype(int)
        quantized_Cr_dct = np.round(Cr_dct / Q_CbCr_with_tile).astype(int)
        
        gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1])
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(np.log(np.abs(quantized_Y_dct) + 0.0001), cmap=gray_colormap)
        ax1.set_title("Y DCT Quantizado")
        
        ax2.imshow(np.log(np.abs(quantized_Cb_dct) + 0.0001), cmap=gray_colormap)
        ax2.set_title("Cb DCT Quantizado")

        ax3.imshow(np.log(np.abs(quantized_Cr_dct) + 0.0001), cmap=gray_colormap)
        ax3.set_title("Cr DCT Quantizado")

        plt.show()

    return quantized_Y_dct, quantized_Cb_dct, quantized_Cr_dct

def inverse_quantized_dct_coefficients_8x8(quantized_Y_dct, quantized_Cb_dct, quantized_Cr_dct):
    qualities = np.array([10, 25, 50, 75, 100])
    
    Q_Y = np.array([[16,  11,  10,  16,  24,  40,  51,  61],
                    [12,  12,  14,  19,  26,  58,  60,  55],
                    [14,  13,  16,  24,  40,  57,  69,  56],
                    [14,  17,  22,  29,  51,  87,  80,  62],
                    [18,  22,  37,  56,  68, 109, 103,  77],
                    [24,  35,  55,  64,  81, 104, 113,  92],
                    [49,  64,  78,  87, 103, 121, 120, 101],
                    [72,  92,  95,  98, 112, 100, 103,  99]])
    
    Q_Y_with_tile = np.tile(Q_Y, (int(len(quantized_Y_dct)/8), int(len(quantized_Y_dct[0])/8)))

    Q_CbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])
    
    Q_CbCr_with_tile = np.tile(Q_CbCr, (int(len(quantized_Cb_dct)/8), int(len(quantized_Cb_dct[0])/8)))
    
    for i in qualities:
        
        Qs_Y = quality_factor(Q_Y_with_tile, i)
        Qs_CbCr = quality_factor(Q_CbCr_with_tile, i)
        
        #upcast to float
        
        Y_dct = quantized_Y_dct * Qs_Y
        Cb_dct = quantized_Cb_dct * Qs_CbCr
        Cr_dct = quantized_Cr_dct * Qs_CbCr
        
        gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1])
        fig = plt.figure()

        # Y DCT
        fig.add_subplot(1, 3, 1)
        plt.title("Y DCT")
        plt.imshow(np.log(np.abs(Y_dct) + 0.0001), cmap=gray_colormap)
        plt.colorbar(shrink=0.5)

        # Cb DCT
        fig.add_subplot(1, 3, 2)
        plt.title("Cb DCT")
        plt.imshow(np.log(np.abs(Cb_dct) + 0.0001), cmap=gray_colormap)
        plt.colorbar(shrink=0.5)

        # Cr DCT
        fig.add_subplot(1, 3, 3)
        plt.title("Cr DCT")
        plt.imshow(np.log(np.abs(Cr_dct) + 0.0001), cmap=gray_colormap)
        plt.colorbar(shrink=0.5)

        plt.subplots_adjust(wspace=0.5)
    
    plt.show()

    return Y_dct, Cb_dct, Cr_dct

def quantized_dct_coefficients_8x8_50(Y_dct, Cb_dct, Cr_dct):
    Q_Y = np.array([[16,  11,  10,  16,  24,  40,  51,  61],
                    [12,  12,  14,  19,  26,  58,  60,  55],
                    [14,  13,  16,  24,  40,  57,  69,  56],
                    [14,  17,  22,  29,  51,  87,  80,  62],
                    [18,  22,  37,  56,  68, 109, 103,  77],
                    [24,  35,  55,  64,  81, 104, 113,  92],
                    [49,  64,  78,  87, 103, 121, 120, 101],
                    [72,  92,  95,  98, 112, 100, 103,  99]])
    
    Q_Y_with_tile = np.tile(Q_Y, (int(len(Y_dct)/8), int(len(Y_dct[0])/8)))

    Q_CbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])
    
    Q_CbCr_with_tile = np.tile(Q_CbCr, (int(len(Cb_dct)/8), int(len(Cb_dct[0])/8)))
    
    Qs_Y = quality_factor(Q_Y_with_tile, 50)
    Qs_CbCr = quality_factor(Q_CbCr_with_tile, 50)
    
    quantized_Y_dct = np.round(Y_dct / Qs_Y).astype(int)
    quantized_Cb_dct = np.round(Cb_dct / Qs_CbCr).astype(int)
    quantized_Cr_dct = np.round(Cr_dct / Qs_CbCr).astype(int)
    
    return quantized_Y_dct, quantized_Cb_dct, quantized_Cr_dct

def inverse_quantized_dct_coefficients_8x8_50(quantized_Y_dct, quantized_Cb_dct, quantized_Cr_dct):
    Q_Y = np.array([[16,  11,  10,  16,  24,  40,  51,  61],
                    [12,  12,  14,  19,  26,  58,  60,  55],
                    [14,  13,  16,  24,  40,  57,  69,  56],
                    [14,  17,  22,  29,  51,  87,  80,  62],
                    [18,  22,  37,  56,  68, 109, 103,  77],
                    [24,  35,  55,  64,  81, 104, 113,  92],
                    [49,  64,  78,  87, 103, 121, 120, 101],
                    [72,  92,  95,  98, 112, 100, 103,  99]])
    
    Q_Y_with_tile = np.tile(Q_Y, (int(len(quantized_Y_dct)/8), int(len(quantized_Y_dct[0])/8)))

    Q_CbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])
    
    Q_CbCr_with_tile = np.tile(Q_CbCr, (int(len(quantized_Cb_dct)/8), int(len(quantized_Cb_dct[0])/8)))
    
    Qs_Y = quality_factor(Q_Y_with_tile, 50)
    Qs_CbCr = quality_factor(Q_CbCr_with_tile, 50)
    
    Y_dct = quantized_Y_dct * Qs_Y
    Cb_dct = quantized_Cb_dct * Qs_CbCr
    Cr_dct = quantized_Cr_dct * Qs_CbCr
    
    return Y_dct, Cb_dct, Cr_dct
    
    
def coefficients_dc(dc, blocks):
    gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1])
    diff = dc.copy()
    for i in range(0, len(dc), blocks):
        for j in range(0, len(dc[0]), blocks):
            if j == 0:
                if i != 0:
                    diff[i][j] = dc[i][j] - dc[i-blocks][len(dc[0])-blocks-1]
            else:
                diff[i][j] = dc[i][j] - dc[i][j-blocks]
    return diff


def inverse_coefficients_dc(diff, blocks):
    gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1])
    dc = diff.copy()
    for i in range(0, len(diff), blocks):
        for j in range(0, len(diff[0]), blocks):
            if j == 0:
                if i != 0:
                    dc[i][j] = dc[i -blocks][len(diff[0])-blocks-1] + diff[i][j]
            else:
                dc[i][j] = dc[i][j-blocks] + diff[i][j]
    plt.figure()
    plt.imshow(np.log(np.abs(dc) + 0.0001), cmap=gray_colormap)
    plt.title("Ex.9._inverso")
    plt.show()
    return dc

def MSE(original_image, recovered_image):
    mse = np.sum((original_image.astype(float) - recovered_image.astype(float)) ** 2)
    mse /= float(original_image.shape[0] * original_image.shape[1])
    return mse


def RMSE(mse):
    rmse = math.sqrt(mse)
    return rmse


def SNR(original_image, mse):
    P = np.sum(original_image.astype(float) ** 2)
    P /= float(original_image.shape[0] * original_image.shape[1])
    snr = 10 * math.log10(P/mse)
    return snr


def PSNR(mse, original_image):
    original = original_image.astype(float)
    max_ = np.max(original) ** 2
    psnr = 10 * math.log10(max_/mse)
    return psnr


def encoder(img,color1, color2, colormap):
    #3 e 4
    
    img_original = read_image(img)
    colormap = colormap_function(colormap, color1, color2)
    draw_plot(img_original, colormap)
    img_padded,nl,nc = padding(img)
    R_p, G_p, B_p = rgb_components(img_padded)
    show_rgb(R_p, G_p, B_p)
    
    #5
    
    Y, Cb, Cr, ycbcr = RGB_toYCbCr(R_p, G_p, B_p, img_padded)
    showCanals(ycbcr) 
    
    #6
    
    Y_d, Cb_d, Cr_d = downsampling(Y, Cb, Cr,1)
    
    #7
    
    Y_dct, Y_dct_log = dct(Y_d, "Y_d", 8, 8, 1)
    Cb_dct, Cb_dct_log = dct(Cb_d, "Cb_d", 8, 8, 1)
    Cr_dct, Cr_dct_log = dct(Cr_d, "Cr_d", 8, 8, 1)
    
    #8
    
    quantized_Y_dct, quantized_Cb_dct, quantized_Cr_dct = quantized_dct_coefficients_8x8_50(Y_dct, Cb_dct, Cr_dct)
    
    #9
    
    diff_Y = coefficients_dc(quantized_Y_dct, 8)
    diff_Cb = coefficients_dc(quantized_Cb_dct, 8)
    diff_Cr = coefficients_dc(quantized_Cr_dct, 8)
    
    return nl,nc, diff_Y, diff_Cb, diff_Cr, img_original
    
def decoder(nl,nc, diff_Y, diff_Cb, diff_Cr, img_original):
    
    #9
    
    quantized_Y_dct = inverse_coefficients_dc(diff_Y, 8)
    quantized_Cb_dct = inverse_coefficients_dc(diff_Cb, 8)
    quantized_Cr_dct = inverse_coefficients_dc(diff_Cr, 8)
    
    #8
    
    Y_dct, Cb_dct, Cr_dct = inverse_quantized_dct_coefficients_8x8_50(quantized_Y_dct, quantized_Cb_dct, quantized_Cr_dct)
   
    #7
    
    Y_d, Y_d_block = dct_inverse(Y_dct, 8, 8)
    Cb_d, Cb_d_block = dct_inverse(Cb_dct, 8, 8)
    Cr_d, Cr_d_block = dct_inverse(Cr_dct, 8, 8)
    
    #6
        
    Y_d, Cb_u, Cr_u = upsampling(Y_d, Cb_d, Cr_d, 1)
    
    #5
   
    R, G, B = YCbCr_to_RGB(Y_d, Cb_u, Cr_u)
    
    #3
    
    matrix_joined_rgb = rgb_components_reverse(R, G, B)
    
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.title("RGB channels joined with padding")
    plt.imshow(matrix_joined_rgb)
    
    #4
    
    img_wp = reverse_padding(matrix_joined_rgb, nl, nc)
    
    
    mse = MSE(img_original, img_wp)
    print("MSE: ", end="")
    print(format(mse, ".3f"))
    rmse = RMSE(mse)
    print("RMSE: ", end="")
    print(format(rmse, ".3f"))
    snr = SNR(img_original, mse)
    print("SNR: ", end="")
    print(format(snr, ".3f"))
    psnr = PSNR(mse, img_original)
    print("PSNR: ", end="")
    print(format(psnr, ".3f"))
    
def main():
    
    nl,nc,diff_Y, diff_Cb, diff_Cr, img_original = encoder("imagens/barn_mountains.bmp",(1, 1, 1),(1, 0, 0),"Red")
    
    decoder(nl,nc, diff_Y, diff_Cb, diff_Cr, img_original)
    
if __name__ == "__main__":
    main()