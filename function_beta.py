import numpy as np
import cv2

#вырезания куска фото
def crop_img(img):

    row_h = int(img.shape[0] * 0.17)
    row_l = int(img.shape[0] * 1.0)
    col_l = int(img.shape[1] * 0.10)
    col_r = int(img.shape[1] * 0.90)

#    row_h = int(img.shape[0] * 0.0)
#    row_l = int(img.shape[0] * 1.0)
#    col_l = int(img.shape[1] * 0.0)
#    col_r = int(img.shape[1] * 1.0)
    
    crop_img = img[row_h:row_l, col_l:col_r]
    return crop_img

#измерение фокуса кадра (реализация Krotkov86)
def TENG(img, ksize=3):
    ''''TENG' algorithm (Krotkov86)'''
    Gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    Gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    FM = Gx*Gx + Gy*Gy
    mn = cv2.mean(FM)[0]

    return mn

def MLOG(img):
    """Implements the MLOG focus measure algorithm.

    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """

    laplas = cv2.Laplacian(img, 3)
    return np.max(cv2.convertScaleAbs(laplas))

#вычисление блюра по всему изображению
def blur_score(img: np.ndarray) -> np.float64:
    return cv2.Laplacian(img, cv2.CV_64F).var()

#среднее по всему изображению
def gray_score(gray_img: np.ndarray) -> float:
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rating = cv2.mean(gray_img)[0] / 255.0
    return rating



#*************************************************************************************
# расчёт текстурных признаков GLCM (Gray-Level Co-Occurrence Matrix) 
#вычисление graycomatrix собственным методом
def my_glcm_numpy(img, distance=50):

    glcm = np.zeros([256,256], dtype=np.int32)
    img_shape_0 = img.shape[0]
    img_shape_1 = img.shape[1]
        
    for i in range(img_shape_0 - distance): # row
        new_coord = i + distance
        for j in range(img_shape_1): # col
            init_val = img[i,j]
            target = img[new_coord, j]
            glcm[init_val,target] += 1
        
    return glcm

#расчет contrast, dissimilarity, homogeneity, energy, correlation
#на основе предварительно вычисленной GLCM матрицы

#функция взята из scikit image

def graycoprops(P, prop='contrast'):
    """Calculate texture properties of a GLCM.

    Compute a feature of a gray level co-occurrence matrix to serve as
    a compact summary of the matrix. The properties are computed as
    follows:

    - 'contrast': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-j)^2`
    - 'dissimilarity': :math:`\\sum_{i,j=0}^{levels-1}P_{i,j}|i-j|`
    - 'homogeneity': :math:`\\sum_{i,j=0}^{levels-1}\\frac{P_{i,j}}{1+(i-j)^2}`
    - 'ASM': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}^2`
    - 'energy': :math:`\\sqrt{ASM}`
    - 'correlation':
        .. math:: \\sum_{i,j=0}^{levels-1} P_{i,j}\\left[\\frac{(i-\\mu_i) \\
                  (j-\\mu_j)}{\\sqrt{(\\sigma_i^2)(\\sigma_j^2)}}\\right]

    Each GLCM is normalized to have a sum of 1 before the computation of
    texture properties.

    .. versionchanged:: 0.19
           `greycoprops` was renamed to `graycoprops` in 0.19.

    Parameters
    ----------
    P : ndarray
        Input array. `P` is the gray-level co-occurrence histogram
        for which to compute the specified property. The value
        `P[i,j,d,theta]` is the number of times that gray-level j
        occurs at a distance d and at an angle theta from
        gray-level i.
    prop : {'contrast', 'dissimilarity', 'homogeneity', 'energy', \
            'correlation', 'ASM'}, optional
        The property of the GLCM to compute. The default is 'contrast'.

    Returns
    -------
    results : 2-D ndarray
        2-dimensional array. `results[d, a]` is the property 'prop' for
        the d'th distance and the a'th angle.

    References
    ----------
    .. [1] M. Hall-Beyer, 2007. GLCM Texture: A Tutorial v. 1.0 through 3.0.
           The GLCM Tutorial Home Page,
           https://prism.ucalgary.ca/handle/1880/51900
           DOI:`10.11575/PRISM/33280`

    Examples
    --------
    Compute the contrast for GLCMs with distances [1, 2] and angles
    [0 degrees, 90 degrees]

    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> g = graycomatrix(image, [1, 2], [0, np.pi/2], levels=4,
    ...                  normed=True, symmetric=True)
    >>> contrast = graycoprops(g, 'contrast')
    >>> contrast
    array([[0.58333333, 1.        ],
           [1.25      , 2.75      ]])

    """

    (num_level, num_level2, num_dist, num_angle) = P.shape
    if num_level != num_level2:
        raise ValueError('num_level and num_level2 must be equal.')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive.')
    if num_angle <= 0:
        raise ValueError('num_angle must be positive.')

    # normalize each GLCM
    P = P.astype(np.float64)
    glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums

    # create weights for specified property
    I, J = np.ogrid[0:num_level, 0:num_level]
    if prop == 'contrast':
        weights = (I - J) ** 2
    elif prop == 'dissimilarity':
        weights = np.abs(I - J)
    elif prop == 'homogeneity':
        weights = 1.0 / (1.0 + (I - J) ** 2)
    elif prop in ['ASM', 'energy', 'correlation']:
        pass
    else:
        raise ValueError(f'{prop} is an invalid property')

    # compute property for each GLCM
    if prop == 'energy':
        asm = np.sum(P**2, axis=(0, 1))
        results = np.sqrt(asm)
    elif prop == 'ASM':
        results = np.sum(P**2, axis=(0, 1))
    elif prop == 'correlation':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.sum(I * P, axis=(0, 1))
        diff_j = J - np.sum(J * P, axis=(0, 1))

        std_i = np.sqrt(np.sum(P * (diff_i) ** 2, axis=(0, 1)))
        std_j = np.sqrt(np.sum(P * (diff_j) ** 2, axis=(0, 1)))
        cov = np.sum(P * (diff_i * diff_j), axis=(0, 1))

        # handle the special case of standard deviations near zero
        mask_0 = std_i < 1e-15
        mask_0[std_j < 1e-15] = True
        results[mask_0] = 1

        # handle the standard case
        mask_1 = ~mask_0
        results[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
    elif prop in ['contrast', 'dissimilarity', 'homogeneity']:
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.sum(P * weights, axis=(0, 1))

    return results

def contrast_2_score(glcm) -> np.float64:
    contrast = graycoprops(glcm, 'contrast')
    return contrast.item()

def dissimilarity_score(glcm) -> np.float64:
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    return dissimilarity.item()

def homogeneity_score(glcm) -> np.float64:
    homogeneity = graycoprops(glcm, 'homogeneity')
    return homogeneity.item()

def energy_score(glcm) -> np.float64:
    energy = graycoprops(glcm, 'energy')
    return energy.item()

def corr_score(glcm) -> np.float64:
    correlation = graycoprops(glcm, 'correlation')
    return correlation.item()


#*************************************************************************************

#подсчет количество квадратов где:
#std больше порога
#среднее больше среднего по всему фото
#среднее больше 120 (выявлено экспериментальным путём). Это же расчет glare, то есть подсчет квадратов где "сильно ярко"
def get_std_square(img_gray):
    n = 8
    count = 0
    count_more_mean = 0
    count_more_120 = 0
    img_gray_mean = img_gray.mean()
    for i in range(0, img_gray.shape[0], n):
        for j in range(0, img_gray.shape[1], n):
            crop = img_gray[i:i+n,j:j+n]
            crop_std = crop.std()
            crop_mean = crop.mean()
            if crop_std > 12.0:
                count += 1
            if crop_mean > img_gray_mean:
                count_more_mean += 1
            if crop_mean > 120:
                count_more_120 += 1

    return count, count_more_mean, count_more_120

# расчет шумов
def calculate_noise_metrics(img_gray):

    # Apply Gaussian blur to the grayscale image
    blurred_image = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # Calculate the noise by subtracting the blurred image from the original grayscale image
    noise = img_gray - blurred_image
    # Calculate the mean and standard deviation of the noise
    mean_noise = np.mean(noise)
    std_noise = np.std(noise)

    return mean_noise, std_noise

#расчет процента пикселей с насыщенностью >= порога
def is_valid(image):

    # Convert image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate histogram of saturation channel
    s = cv2.calcHist([image], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.05
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])

    return s_perc

#**************************************************************
#---------------------хроматические признаки--------------------
#**************************************************************

def calculate_colorfulness(image):
    """
    Вычисление цветности (colorfulness) изображения.
    Метрика основана на работе Hasler and Süsstrunk (2003).
    """
    # Разделение каналов
    (B, G, R) = cv2.split(image.astype("float"))

    # Вычисление rg и yb
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)

    # Средние значения и стандартные отклонения
    rg_mean, rg_std = np.mean(rg), np.std(rg)
    yb_mean, yb_std = np.mean(yb), np.std(yb)

    # Цветность
    colorfulness = rg_std + yb_std + 0.3 * (rg_mean + yb_mean)
    return colorfulness

def calculate_saturation(image):
    """
    Вычисление средней насыщенности изображения.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()
    return saturation

def calculate_white_balance(image):
    """
    Оценка баланса белого через отклонение от серого.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_avg = gray.mean()
    deviation = np.abs(image - gray_avg).mean()
    return deviation


#**************************************************************
#---------------------признаки на основе fft--------------------
#**************************************************************
def fft_features(img_gray):
    dft = cv2.dft(np.float32(img_gray),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    dft_shift[dft_shift == 0.0] = 1.0
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    
    #print(magnitude_spectrum.shape)
    #magn_spectr = cv2.resize(magnitude_spectrum, (256, 256), interpolation = cv2.INTER_LINEAR)
    #среднее из ВЧ части по главному направлению

    #lf_mean = magnitude_spectrum[30:34,63:81].mean()
    #mf_mean = magnitude_spectrum[30:34,80:96].mean()
    hf_mean = magnitude_spectrum[30:34,95:].mean()
    #hf_std = magnitude_spectrum[0:16,96:].mean()
    #hf_std = magnitude_spectrum[0:12,63:65].mean()

    return hf_mean#, hf_std

#**************************************************************
#------Helmholtz-Kohlrausch (HK) features признаки------------
#**************************************************************
def helmholtz_kohlrausch_fairchild_trig(image_bgr: np.ndarray):
    """
    Compute Helmholtz-Kohlrausch (HK) features using Fairchild's
    hue-dependent factor f(h) with trigonometric terms.

    References the formula:
      f(h) = -0.160*cos(h) + 0.132*cos(2h)
             -0.405*sin(h) + 0.080*sin(2h)
             +0.792

    where h is in radians.

    Returns two maps:
      1) Adjusted Lightness L_adj
      2) Perceived Brightness B (HK-like)

    :param image_bgr: Input image in BGR (uint8 or float).
    :return: (L_adj_map, B_map) - two float32 single-channel images.
    """

    # ----------------------------------------------------------------
    # 1) Convert to float if needed and normalize to [0,1]
    # ----------------------------------------------------------------
    if image_bgr.dtype not in (np.float32, np.float64):
        image_bgr = image_bgr.astype(np.float32) / 255.0

    # ----------------------------------------------------------------
    # 2) BGR -> Lab
    #    OpenCV's Lab typically: L in [0..100], a,b in [-128..127]
    # ----------------------------------------------------------------
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    L_channel, a, b = cv2.split(image_lab)

    # ----------------------------------------------------------------
    # 3) Lab -> LCh
    #    C = sqrt(a^2 + b^2)
    #    h = atan2(b, a)  (range: -pi..pi)
    #    For trig usage, we do NOT convert to degrees if f(h) expects rad.
    # ----------------------------------------------------------------
    chroma = np.sqrt(a * a + b * b)
    hue_angle = np.arctan2(b, a)  # in radians: -pi..pi

    # If you want h in [0..2*pi) for consistent trig interpretation:
    hue_angle = np.where(hue_angle < 0, hue_angle + 2.0*np.pi, hue_angle)

    # ----------------------------------------------------------------
    # 4) Fairchild's hue-dependent factor f(h) in radians
    #    f(h) = -0.160*cos(h) + 0.132*cos(2*h)
    #           -0.405*sin(h) + 0.080*sin(2*h)
    #           +0.792
    # ----------------------------------------------------------------
    f_hue = (
            -0.160 * np.cos(hue_angle)
            + 0.132 * np.cos(2.0 * hue_angle)
            - 0.405 * np.sin(hue_angle)
            + 0.080 * np.sin(2.0 * hue_angle)
            + 0.792
    )

    # ----------------------------------------------------------------
    # 5) Adjusted Lightness, L_adj
    #    One possible approach:
    #        L_adj = L * f(h)
    # ----------------------------------------------------------------
    L_HK = L_channel * f_hue

    # ----------------------------------------------------------------
    # 6) Perceived Brightness (HK-like)
    #    B = L_adj * [1 + alpha * (C^gamma)]
    #    Using typical placeholder values alpha=0.02, gamma=1.7
    # ----------------------------------------------------------------
    alpha = 0.02
    gamma = 1.7
    B_HK = L_HK * (1.0 + alpha * (chroma ** gamma))

    # ----------------------------------------------------------------
    # 7) Ensure float32 outputs
    # ----------------------------------------------------------------
    L_HK = L_HK.astype(np.float32)
    B_HK = B_HK.astype(np.float32)

    features = {
        "mean_L": np.mean(L_channel),
        "std_L": np.std(L_channel),
        "mean_C": np.mean(chroma),
        "std_C": np.std(chroma),
        "mean_h": np.mean(hue_angle),
        "std_h": np.std(hue_angle),
        "mean_L_HK": np.mean(L_HK),
        "std_L_HK": np.std(L_HK),
        "mean_B_HK": np.mean(B_HK),
        "std_B_HK": np.std(B_HK),
    }

    return features

# Проверка на монохромность
def flag_monochrome(img, threshold=10):
    """
    Проверяет, является ли изображение монохромным (чёрно-белым или градации серого).
    
    Параметры:
    - image_path: путь к изображению
    - threshold: пороговое значение для определения монохромности (по умолчанию 10)
    
    Возвращает:
    - True, если изображение монохромное, иначе False
    """
    
    # Если изображение уже в градациях серого (1 канал)
    if len(img.shape) == 2:
        return 1
    
    # Разделяем изображение на каналы BGR
    b, g, r = cv2.split(img)
    
    # Вычисляем разницу между каналами
    diff_rg = np.mean(np.abs(r - g))
    diff_rb = np.mean(np.abs(r - b))
    diff_gb = np.mean(np.abs(g - b))
    
    # Если средние разницы между каналами меньше порога - изображение монохромное
    if diff_rg < threshold and diff_rb < threshold and diff_gb < threshold:
        return 1
    else:
        return 0

#получение среднего по каждому каналу
def get_mean_each_channel(img):
    b, g, r = cv2.split(img)
    return b.mean(), g.mean(), r.mean()


