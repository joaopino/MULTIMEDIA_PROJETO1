import cv2
import numpy as np

# Lê a imagem BMP
img = cv2.imread('imagem.bmp')

# Converte a imagem para o espaço de cores YCbCr
img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Define os parâmetros de qualidade desejados (entre 0 e 100)
qualities = [10, 25, 50, 75, 100]

for quality in qualities:
    # Realiza a quantização dos coeficientes DCT de acordo com o parâmetro de qualidade
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    Q = np.multiply(Q, quality / 50)
    Q = np.round(Q)
    Q[Q == 0] = 1

    # Divide a imagem em blocos de 8x8 pixels e aplica a DCT em cada bloco
    rows, cols, _ = img_ycc.shape
    new_rows = np.uint32(np.ceil(rows / 8) * 8)
    new_cols = np.uint32(np.ceil(cols / 8) * 8)
    padded_img = np.zeros((new_rows, new_cols, 3), dtype=np.uint8)
    padded_img[:rows, :cols] = img_ycc
    im_blocks = [np.zeros((8, 8, 3), dtype=np.float32) for _ in range(new_rows // 8 * new_cols // 8)]
    i = 0
    for row in range(0, new_rows, 8):
        for col in range(0, new_cols, 8):
            im_blocks[i] = np.float32(padded_img[row:row + 8, col:col + 8])
            i += 1
    im_blocks = np.array(im_blocks)

    im_blocks_dct = np.zeros(im_blocks.shape, dtype=np.float32)
    for i in range(im_blocks.shape[0]):
        im_blocks_dct[i] = cv2.dct(im_blocks[i] - 128)

    # Quantiza os coeficientes DCT
    im_blocks_quantized = np.zeros(im_blocks_dct.shape, dtype=np.int16)
    for i in range(im_blocks.shape[0]):
        im_blocks_quantized[i] = np.round(im_blocks_dct[i] / Q)

# Codifica os coeficientes quantizados
    im_blocks_encoded = []
    for i in range(im_blocks_quantized.shape[0]):
        block_flat = im_blocks_quantized[i].reshape(-1)
        nonzero_indices = np.nonzero(block_flat)[0]
        if len(nonzero_indices) == 0:
            # Caso o bloco seja totalmente nulo, adiciona um marcador de bloco nulo
            im_blocks_encoded.append(('Z', 0))
        else:
            # Adiciona o marcador de início de bloco não nulo
            im_blocks_encoded.append(('N', len(nonzero_indices)))
            # Adiciona os coeficientes não nulos do bloco
            for j in nonzero_indices:
                im_blocks_encoded.append(('V', block_flat[j]))

    # Concatena os blocos quantizados em um único array
for i in range(im_blocks_quantized.shape[0]):
    block_flat = im_blocks_quantized[i].reshape(-1)
    nonzero_indices = np.nonzero(block_flat)[0]
    if len(nonzero_indices) == 0:
        # Caso o bloco seja totalmente nulo, adiciona um marcador de bloco nulo
        im_blocks_encoded.append(('Z', 0))
    else:
        # Adiciona o marcador de início de bloco não nulo
        im_blocks_encoded.append(('N', len(nonzero_indices)))
        # Adiciona os coeficientes não nulos do bloco
        for j in nonzero_indices:
            im_blocks_encoded.append(('V', block_flat[j]))

# Decodifica os coeficientes quantizados
im_blocks_decoded = np.zeros(im_blocks_quantized.shape, dtype=np.float32)
i = 0
for j in range(len(im_blocks_encoded)):
    if im_blocks_encoded[j][0] == 'Z':
        # Se o bloco for nulo, preenche-o com zeros
        im_blocks_decoded[i] = np.zeros((8, 8, 3), dtype=np.float32)
        i += 1
    elif im_blocks_encoded[j][0] == 'N':
        # Se o bloco for não nulo, cria um novo bloco e adiciona os coeficientes
        block_flat = np.zeros((8, 8, 3), dtype=np.float32).reshape(-1)
        nonzero_indices = []
        for k in range(im_blocks_encoded[j][1]):
            nonzero_indices.append(im_blocks_encoded[j + k + 1][1])
        for k in range(im_blocks_encoded[j][1]):
            block_flat[nonzero_indices[k]] = im_blocks_encoded[j + k + 1][1]
        im_blocks_decoded[i] = np.multiply(cv2.idct(block_flat.reshape((8, 8, 3))) + 128, Q)
        i += 1

# Junta os blocos decodificados para reconstruir a imagem
padded_img_reconstructed = np.zeros(padded_img.shape, dtype=np.float32)
i = 0
for row in range(0, new_rows, 8):
    for col in range(0, new_cols, 8):
        padded_img_reconstructed[row:row + 8, col:col + 8] = im_blocks_decoded[i]
        i += 1

# Converte a imagem reconstruída de volta para o espaço de cores BGR
img_reconstructed_ycc = padded_img_reconstructed[:rows, :cols]
img_reconstructed_bgr = cv2.cvtColor(img_reconstructed_ycc, cv2.COLOR_YCrCb2BGR)

# Salva a imagem reconstruída com o nome "imagem_qualidade.bmp", onde "qualidade" é o parâmetro de qualidade atual
cv2.imwrite(f'imagem_{quality}.bmp', img_reconstructed_bgr)