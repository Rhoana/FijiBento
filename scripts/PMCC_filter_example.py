import sys
import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter
import pylab

FAIL_PMCC_SCORE_TOO_LOW = 0
FAIL_PMCC_ON_EDGE = 1
FAIL_PMCC_CURVATURE_TOO_HIGH = 2
FAIL_PMCC_MAXRATIO_TOO_HIGH = 3
FAIL_PMCC_NOT_LOCALIZED = 4

# TODO: smoothing?

def PMCC_match(image, template, min_correlation=0.6, maximal_curvature_ratio=10, maximal_ROD=0.9, output=[None]):
    if output[0] is None:
        output[0] = (np.empty([sz - tz + 1 for sz, tz in zip(image.shape, template.shape)],
                              dtype=np.float32),
                     np.empty([sz - tz + 1 for sz, tz in zip(image.shape, template.shape)],
                              dtype=np.float32))

    correlation_image = output[0][0]
    maximum_image = output[0][1]

    # compute the correlation image
    cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, correlation_image)

    # find local maxima
    maximum_filter(correlation_image, size=3, output=maximum_image)
    maxima_mask = (correlation_image == maximum_image)
    maxima_values = correlation_image[maxima_mask]
    maxima_values.sort()

    if maxima_values[-1] < min_correlation:
        return None, FAIL_PMCC_SCORE_TOO_LOW, maxima_values[-1]

    # TrakEM2 code uses (1 + 2nd_best) / (1 + best) for this test...?
    if (maxima_values.size > 1) and (maxima_values[-2] / maxima_values[-1] > maximal_ROD):
        return None, FAIL_PMCC_MAXRATIO_TOO_HIGH, maxima_values[-2] / maxima_values[-1]

    # find the maximum location
    mi, mj = np.unravel_index(np.argmax(correlation_image), correlation_image.shape)
    if (mi == 0) or (mj == 0) or (mi == correlation_image.shape[0] - 1) or (mj == correlation_image.shape[1] - 1):
        return None, FAIL_PMCC_ON_EDGE, None

    # extract pixels around maximum
    [[c00, c01, c02],
     [c10, c11, c12],
     [c20, c21, c22]] = correlation_image[(mi - 1):(mi + 2),
                                          (mj - 1):(mj + 2)]

    dx = (c12 - c10) / 2.0
    dy = (c21 - c01) / 2.0
    dxx = c10 - c11 - c11 + c12
    dyy = c01 - c11 - c11 + c21
    dxy = (c22 - c20 - c02 + c00) / 4.0

    det = dxx * dyy - dxy * dxy
    trace = dxx + dyy
    if (det <= 0) or (trace * trace / det > maximal_curvature_ratio):
        return None, FAIL_PMCC_CURVATURE_TOO_HIGH, trace * trace / det

    # localize by Taylor expansion
    # invert Hessian
    ixx = dyy / det
    ixy = -dxy / det
    iyy = dxx / det

    # calculate offset
    ox = -ixx * dx - ixy * dy
    oy = -ixy * dx - iyy * dy

    if abs(ox) >= 1 or abs(oy) >= 1:
        return None, FAIL_PMCC_NOT_LOCALIZED, (ox, oy)

    return True, (mi + oy, mj + ox), None

if __name__ == '__main__':
    # template = cv2.imread(sys.argv[1], 0)  # flags=0 -> grayscale
    image_1 = 255 - cv2.imread(sys.argv[1], 0)
    image_2 = 255 - cv2.imread(sys.argv[2], 0)

    rows, cols = image_2.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -180 * np.pi * 0.006, 1)
    image_2 = cv2.warpAffine(image_2, M, (cols,rows))
    
    for i in range(3):
        image_1 = cv2.resize(image_1, ((image_1.shape[0] / 2, image_1.shape[1] / 2)), interpolation=cv2.INTER_CUBIC)
        image_2 = cv2.resize(image_2, ((image_2.shape[0] / 2, image_2.shape[1] / 2)), interpolation=cv2.INTER_CUBIC)

    print image_1.shape

    for i in range(100, image_1.shape[0], 5):
        for j in range(0, image_1.shape[1], 5):
            template = image_1[i:, j:][:25, :25].copy()
            result, location, value = PMCC_match(image_2, template, min_correlation=0.)
            if result:
                pass
            #print location
            else:
                if location != 3:
                    print "     No match", location, value
