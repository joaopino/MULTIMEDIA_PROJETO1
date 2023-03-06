import matplotlib.pyplot as plt
import matplotlib.colors as clr

import cv2
import numpy as np

import scipy.fftpack as fft

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
    plt.show()
    plt.title("Imagem original")


#3.4

def rgb_components(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return R, G, B

def rgb_components_reverse(R, G, B):
    matrix_inverted = np.zeros((len(R), len(R[0]), 3), dtype=np.uint8)
    matrix_inverted[:, :, 0] = R
    matrix_inverted[:, :, 1] = G
    matrix_inverted[:, :, 2] = B

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
    plt.show()
    plt.title("Imagem com padding")
    
    return imgp,nl,nc

def reverse_padding(image_array, nl_original, nc_original):
    # Extract the unpadded image
    unpadded_image = image_array[:nl_original, :nc_original ]
    
    plt.figure()
    plt.imshow(unpadded_image)
    plt.show()
    plt.title("Imagem sem padding(original)")
    
#5

def RGB_toYCbCr(ImageFile):
    ycbr = np.dot(ImageFile,ycbcr_matrix.T)
    
    R= ycbr[:,:,0] + 128
    G= ycbr[:,:,1] + 128
    B= ycbr[:,:,2] + 128
    
    return ycbr

def YCbCr_to_RGB(ycbcr):
    inverse_transform = np.linalg.inv(ycbcr_matrix)
    
    G= ycbcr[:,:,1] - 128
    B= ycbcr[:,:,2] - 128
    
    rgb = np.dot(ycbcr, inverse_transform.T)
    
    rgb[rgb<0] = 0
    rgb[rgb>255] = 255
    normalized_rgb = np.round(rgb).astype(np.uint8)
    
    plt.figure()
    plt.imshow(normalized_rgb,None)
    plt.show()
    plt.title("Imagem original")

def showCanals(ycbcr_image):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plota o canal Y com colormap gray
    ax1.imshow(ycbcr_image[:, :, 0], cmap='gray')
    ax1.set_title("Canal Y")

    # Plota o canal Cb com colormap jet
    ax2.imshow(ycbcr_image[:, :, 1], cmap='gray')
    ax2.set_title("Canal Cb")

    # Plota o canal Cr com colormap jet
    ax3.imshow(ycbcr_image[:, :, 2], cmap='gray')
    ax3.set_title("Canal Cr")
    
    plt.show()
    plt.title("Mostra os canais Y, Cb, Cr da imagem)")
    
#6
    
def downsampling(img, type):
    
    Y = cv2.pyrDown(img[:,:,0])
    Cb = cv2.pyrDown(img[:,:,1])
    Cr = cv2.pyrDown(img[:,:,2])
    
    # 4:2:0
    if type == 0:
        scaleX = 0.5
        scaleY = 0.5
        Cb_d = cv2.resize(Cb, None, fx=scaleX, fy=scaleY,interpolation=cv2.INTER_LINEAR)
        Cr_d = cv2.resize(Cr, None, fx=scaleX, fy=scaleY,interpolation=cv2.INTER_LINEAR)

    # 4:2:2
    else:
        scaleX = 0.5
        scaleY = 1
        Cb_d = cv2.resize(Cb, None, fx=scaleX, fy=scaleY,interpolation=cv2.INTER_LINEAR)
        Cr_d = cv2.resize(Cr, None, fx=scaleX, fy=scaleY,interpolation=cv2.INTER_LINEAR)

    gray_colormap = colormap_function("Gray", [0, 0, 0], [1, 1, 1])
    
    fig = plt.figure()
    
    fig.add_subplot(3, 2, 3)
    plt.title("Cb - Downsampling 4:2:2")
    plt.imshow(Cb_d, cmap=gray_colormap)

    fig.add_subplot(3, 2, 4)
    plt.title( "Cr - Downsampling 4:2:2")
    plt.imshow(Cr_d, cmap=gray_colormap)

    plt.subplots_adjust(hspace=0.5)

    return Y, Cb_d, Cr_d

def upsampling(Y_d, Cb_d, Cr_d, type):
    
    if type == 0:
        scaleX = 0.5
        scaleY = 0.5
        stepX = int(scaleX)
        stepY = int(scaleY)
        
        Cb_u = cv2.resize(Cb_d, None, fx=stepX, fy=stepY,interpolation=cv2.INTER_LINEAR)
        Cr_u = cv2.resize(Cr_d, None, fx=stepX, fy=stepY,interpolation=cv2.INTER_LINEAR)
    else:
        scaleX = 0.5
        scaleY = 1
        stepX = int(scaleX)
        stepY = int(scaleY)
        
        Cb_u = cv2.resize(Cb_d, None, fx=stepX, fy=stepY,interpolation=cv2.INTER_LINEAR)
        Cr_u = cv2.resize(Cr_d, None, fx=stepX, fy=stepY,interpolation=cv2.INTER_LINEAR)
    
    gray_colormap = colormap_function("Gray", [0, 0, 0], [1, 1, 1])
    
    fig = plt.figure()
    
    fig.add_subplot(3, 2, 3)
    plt.title("Cb - Upsampling 4:2:2")
    plt.imshow(Cb_u, cmap=gray_colormap)

    fig.add_subplot(3, 2, 4)
    plt.title( "Cr - Upsampling 4:2:2")
    plt.imshow(Cr_u, cmap=gray_colormap)

    plt.subplots_adjust(hspace=0.5)

    return Y_d, Cb_u, Cr_u
    
def dct(channel, blocks, channel_0):
    channel_dct = np.zeros(channel.shape)
    channel_dct_log = np.zeros(channel.shape)
    channel_dct_block = np.zeros(channel.shape)

    for i in range(0, len(channel), blocks):
        for j in range(0, len(channel[0]), channel_0):
            channel_dct = fft.dct(fft.dct(channel, norm="ortho").T, norm="ortho").T
            channel_dct_block[i:i+blocks, j:j+channel_0] = fft.dct(fft.dct(channel[i:i+blocks, j:j+channel_0], norm="ortho").T, norm="ortho").T
            channel_aux = fft.dct(fft.dct(channel[i:i+blocks, j:j+channel_0], norm="ortho").T, norm="ortho").T
            channel_dct_log[i:i+blocks, j:j +channel_0] = np.log(np.abs(channel_aux) + 0.0001)
            
    plt.imshow(channel_dct, cmap='gray')
    plt.title("DCT do canal" + channel + "completo")
    
    plt.imshow(channel_dct_log, cmap='gray')
    plt.title("DCT do canal" + channel + "completo com funçaõ logaritmica")
    
    plt.imshow(channel_dct_block, cmap='gray')
    plt.title("DCT do canal" + channel + 'em blocos' + blocks + "x" + blocks)
    
    plt.show()
    return channel_dct, channel_dct_block, channel_dct_log

def dct_inverse(channel_dct,  blocks, channel_0):
    channel_d = np.zeros(channel_dct.shape)
    channel_d_block = np.zeros(channel_dct.shape)

    for i in range(0, len(channel_dct), blocks):
        for j in range(0, len(channel_dct[0]), channel_0):
            chanel_d = fft.idct(fft.idct(channel_dct, norm="ortho").T, norm="ortho").T
            channel_d_block[i:i+blocks, j:j+channel_0] = fft.idct(fft.idct( channel_dct[i:i+blocks, j:j+channel_0], norm="ortho").T, norm="ortho").T

    plt.imshow(channel_d, cmap='gray')
    plt.title("DCT inverso canal completo")
    
    plt.imshow(channel_d, cmap='gray')
    plt.title("DCT inverso canal em blocos")
    
    plt.show()

    return channel_d, channel_d_block

#8

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
        if(i >= 50):
            sf = (100 - i) / 50
        else:
            sf = 50 / i

        if(sf != 0):
            Qs_Y = np.round(Q_Y_with_tile * sf)
        else:
            Qs_Y = np.ones(Q_Y_with_tile.shape, dtype=np.uint8)
        
        Qs_Y[Qs_Y > 255] = 255
        Qs_Y[Qs_Y < 1] = 1
            
        if(i >= 50):
            sf = (100 - i) / 50
        else:
            sf = 50/i

        if(sf != 0):
            Qs_CbCr = np.round(Q_CbCr_with_tile * sf)
        else:
            Qs_CbCr = np.ones(Q_CbCr_with_tile.shape, dtype=np.uint8)

        Qs_CbCr[Qs_CbCr > 255] = 255
        Qs_CbCr[Qs_CbCr < 1] = 1
        
        quantized_Y_dct = np.round(Y_dct / Q_Y_with_tile)
        quantized_Cb_dct = np.round(Cb_dct / Q_CbCr_with_tile)
        quantized_Cr_dct = np.round(Cr_dct / Q_CbCr_with_tile)
        
        gray_colormap = colormap_function("gray", [0, 0, 0], [1, 1, 1])
        fig = plt.figure()

        # Y DCT
        fig.add_subplot(1, 3, 1)
        plt.title("Y DCT Cuantizado")
        plt.imshow(np.log(np.abs(quantized_Y_dct) + 0.0001), cmap=gray_colormap)
        plt.colorbar(shrink=0.5)

        # Cb DCT
        fig.add_subplot(1, 3, 2)
        plt.title("Cb DCT Cuantizado")
        plt.imshow(np.log(np.abs(quantized_Cb_dct) + 0.0001), cmap=gray_colormap)
        plt.colorbar(shrink=0.5)

        # Cr DCT
        fig.add_subplot(1, 3, 3)
        plt.title("Cr DCT Cuantizado")
        plt.imshow(np.log(np.abs(quantized_Cr_dct) + 0.0001), cmap=gray_colormap)
        plt.colorbar(shrink=0.5)

        plt.subplots_adjust(wspace=0.5)

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
        if(i >= 50):
            sf = (100 - i) / 50
        else:
            sf = 50/i

        if(sf != 0):
            Qs_Y = np.round(Q_Y_with_tile * sf)
        else:
            Qs_Y = np.ones(Q_Y_with_tile.shape, dtype=np.uint8)
        
        Qs_Y[Qs_Y > 255] = 255
        Qs_Y[Qs_Y < 1] = 1
            
        if(i >= 50):
            sf = (100 - i) / 50
        else:
            sf = 50/i

        if(sf != 0):
            Qs_CbCr = np.round(Q_CbCr_with_tile * sf)
        else:
            Qs_CbCr = np.ones(Q_CbCr_with_tile.shape, dtype=np.uint8)

        Qs_CbCr[Qs_CbCr > 255] = 255
        Qs_CbCr[Qs_CbCr < 1] = 1
        
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

    return Y_dct, Cb_dct, Cr_dct

def encoder(img,color1, color2, colormap):
    
    #3 e 4
    
    img_padded,nl,nc = padding(img)
    colormap = colormap_function(colormap, color1, color2)
    draw_plot(img_padded, colormap)
    R_p, G_p, B_p = rgb_components(img_padded)
    show_rgb(R_p, G_p, B_p)
    
    #5
    
    ycbcr = RGB_toYCbCr(img_padded)
    showCanals(ycbcr) 
    
    #6
    
    Y_d, Cb_d, Cr_d = downsampling(R_p, G_p, B_p,0)
    Y_d, Cb_d, Cr_d = downsampling(R_p, G_p, B_p,1)
    
    #7
    
    Y_dct, Y_dct_log, Y_dct_block = dct(Y_d, 64, 64)
    Cb_dct, Cb_dct_log, Cb_dct_block = dct(Cb_d, 64, 64)
    Cr_dct, Cr_dct_log, Cr_dct_block = dct(Cr_d, 64, 64)
    
    Y_dct, Y_dct_log, Y_dct_block = dct(Y_d, 8, 8)
    Cb_dct, Cb_dct_log, Cb_dct_block = dct(Cb_d, 8, 8)
    Cr_dct, Cr_dct_log, Cr_dct_block = dct(Cr_d, 8, 8)
    
    #8
    
    quantized_Y_dct, quantized_Cb_dct, quantized_Cr_dct = quantized_dct_coefficients_8x8(Y_dct, Cb_dct, Cr_dct)
    
    return img_padded,nl,nc,R_p, G_p, B_p,ycbcr, Y_d, Cb_d, Cr_d, Y_dct, Y_dct_log, Cb_dct, Y_dct_block, Cb_dct_block, Cr_dct_block,  quantized_Y_dct, quantized_Cb_dct, quantized_Cr_dct
    
def decoder(image_array, nl_original, nc_original,R_p, G_p, B_p,ycbcr, Y_d, Cb_d, Cr_d, Y_dct, Cb_dct, Cr_dct, quantized_Y_dct, quantized_Cb_dct, quantized_Cr_dct):
    
    #3
    
    matrix_joined_rgb = rgb_components_reverse(R_p, G_p, B_p)
    
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.title("RGB channels joined with padding")
    plt.imshow(matrix_joined_rgb)
    
    #4
    
    reverse_padding(image_array, nl_original, nc_original)
    
    #5
   
    YCbCr_to_RGB(ycbcr)
    
    #6
        
    upsampling(Y_d, Cb_d, Cr_d,0)
    
    #7
    
    Y_d = dct_inverse(Y_dct, 8, 8)
    Cb_d = dct_inverse(Cb_dct, 8, 8)
    Cr_d = dct_inverse(Cr_dct, 8, 8)
    
    #8
    
    Y_dct, Cb_dct, Cr_dct = inverse_quantized_dct_coefficients_8x8(quantized_Y_dct, quantized_Cb_dct, quantized_Cr_dct)

def main():
    
    padded_image,original_lines,original_columns,R_p, G_p, B_p,ycbcr, Y_dct_block, Cb_dct_block, Cr_dct_block = encoder("imagens\barn_mountains.bmp",(1, 1, 1),(1, 0, 0),"Red")
    
    decoder(padded_image,original_lines,original_columns,R_p, G_p, B_p,ycbcr,Y_dct_block, Cb_dct_block, Cr_dct_block)
    
    
if __name__ == "__main__":
    main()