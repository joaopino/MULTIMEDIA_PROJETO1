import matplotlib.pyplot as plt
import matplotlib.colors as clr

import cv2
import numpy as np



##TODO
# Remover o ciclo do join channels
ycbcr_matrix = np.array([
    [65.481, 128.553, 24.966],
    [-37.797, -74.203, 112.0],
    [112.0, -93.786, -18.214]
])


def get_color_input():
    #A
    color1imput = input("Color 1(A,R,G,B): ")
    #R
    color2imput = input("Color 1(A,R,G,B): ")

    if color1imput == "A":
        color1 = (0, 0, 0)
    if color1imput == "R":
        color1 = (1, 0, 0)
    if color1imput == "G":
        color1 = (0, 1, 0)
    if color1imput == "B":
        color1 = (0, 0, 1)
        
    if color2imput == "A":
        color2 = (0, 0, 0)
    if color2imput == "R":
        color2 = (1, 0, 0)
    if color2imput == "G":
        color2 = (0, 1, 0)
    if color2imput == "B":
        color2 = (0, 0, 1)
    return color1,color2

def encoder(img,color1, color2, colormap):
    img_padded,nl,nc = padding(img)
    colormap = Ex3_2(colormap, color1, color2)
    print(colormap)
    Ex3_3(img_padded, colormap)
    R_p, G_p, B_p = Ex3_4_1(img_padded)
    Ex3_5(R_p, G_p, B_p)
    
    ycbcr = RGB_toYCbCr(img_padded)
    showCanals(ycbcr) 
    YCbCr_to_RGB(ycbcr)
    
    return img_padded,nl,nc,R_p, G_p, B_p,ycbcr
    
def decoder(image_array, nl_original, nc_original,R_p, G_p, B_p,ycbcr):
    matrix_joined_rgb = Ex3_4_2(R_p, G_p, B_p)
    
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.title("RGB channels joined with padding")
    plt.imshow(matrix_joined_rgb)
   
    YCbCr_to_RGB(ycbcr)
    
    reverse_padding(image_array, nl_original, nc_original)
    print("REVERSED")

def Ex3_1(img_path):
    return plt.imread(img_path)

def Ex3_2(colormap_name, color1, color2):
    return clr.LinearSegmentedColormap.from_list(colormap_name, [color1, color2], 256)

def Ex3_3(image, colormap):
    plt.figure()
    plt.imshow(image, colormap)
    plt.show()

def Ex3_4_1(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return R, G, B

def Ex3_4_2(R, G, B):
    rows = len(R)
    cols = len(R[0])
    matrix_inverted = [[None] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            matrix_inverted[i][j] = [R[i][j], G[i][j], B[i][j]]

    return matrix_inverted

def Ex3_5(c_R, c_G, c_B):
    cm_red = Ex3_2("Red", (0, 0, 0), (1, 0, 0))
    cm_green = Ex3_2("Green", (0, 0, 0), (0, 1, 0))
    cm_blue = Ex3_2("Blue", (0, 0, 0), (0, 0, 1))
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

def downsampling(Y, Cb, Cr, fatorCb):
    # 4:2:0
    if fatorCb == 0:
        scaleX = 0.5
        scaleY = 0.5
        Cb_d = cv2.resize(Cb, None, fx=scaleX, fy=scaleY,interpolation=cv2.INTER_NEAREST)
        Cr_d = cv2.resize(Cr, None, fx=scaleX, fy=scaleY,interpolation=cv2.INTER_NEAREST)

    # 4:2:2
    else:
        scaleX = 0.5
        scaleY = 1
        Cb_d = cv2.resize(Cb, None, fx=scaleX, fy=scaleY,interpolation=cv2.INTER_NEAREST)
        Cr_d = cv2.resize(Cr, None, fx=scaleX, fy=scaleY,interpolation=cv2.INTER_NEAREST)

    return Y, Cb_d, Cr_d

def upsampling(Y_d, Cb_d, Cr_d):
    scaleX = 0.5
    scaleY = 1
    stepX = int(scaleX)
    stepY = int(scaleY)
    Cb_u = cv2.resize(Cb_d, None, fx=stepX, fy=stepY,interpolation=cv2.INTER_LINEAR)
    Cr_u = cv2.resize(Cr_d, None, fx=stepX, fy=stepY,interpolation=cv2.INTER_LINEAR)
    
    gray_colormap = Ex3_2("Gray", [0, 0, 0], [1, 1, 1])
    
    fig = plt.figure()
    
    fig.add_subplot(3, 2, 3)
    plt.title("Cb - Downsampling 4:2:2")
    plt.imshow(Cb_d, cmap=gray_colormap)

    fig.add_subplot(3, 2, 4)
    plt.title( "Cr - Downsampling 4:2:2")
    plt.imshow(Cr_d, cmap=gray_colormap)

    plt.subplots_adjust(hspace=0.5)

    return Y_d, Cb_u, Cr_u

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
    
    return imgp,nl,nc

def reverse_padding(image_array, nl_original, nc_original):
    # Extract the unpadded image
    unpadded_image = image_array[:nl_original, :nc_original ]
    
    plt.figure()
    plt.imshow(unpadded_image)
    plt.show()

def RGB_toYCbCr(ImageFile):
    ycbr = np.dot(ImageFile,ycbcr_matrix.T)
    
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


def main():
    
    img_name = input("Image path: ")

    colormap = input("Colormap: ")
    
    color1,color2 = get_color_input()
    
    padded_image,original_lines,original_columns,R_p, G_p, B_p,ycbcr = encoder(img_name,color1,color2,colormap)
    
    decoder(padded_image,original_lines,original_columns,R_p, G_p, B_p,ycbcr)
    #================== Exercício 4
    
    #img,nl,nc = padding("imagens/barn_mountains.bmp")
    #reverse_padding(img,nl,nc)
    
    
    #================== Exercício 5
    
    #ycbcr = RGB_toYCbCr(input("Path to Image to turn to YCbCr: "))
    #showCanals(ycbcr) 
    #YCbCr_to_RGB(ycbcr)
    
    
if __name__ == "__main__":
    main()