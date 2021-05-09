import numpy as np
import cv2
import matplotlib.pyplot as plt

# kernels of filters:
sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

roberts_x = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, -1, 0]])

roberts_y = np.array([[0, 0, 0],
                      [0, 1, -1],
                      [0, 0, 0]])

prewitta_x = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])

prewitta_y = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]])

laplace = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])


# lets use open cv for using filters
def sobel_filter_x(img):
    """using a Sobel horizontal filter for an image"""
    grad_x = cv2.filter2D(img, cv2.CV_32F, sobel_x)
    return grad_x


def sobel_filter_y(img):
    """using a Sobel vertical filter for an image"""
    grad_y = cv2.filter2D(img, cv2.CV_32F, sobel_y)
    return grad_y


def sobel_filter(img):
    """using a Sobel(x+y) filter for an image"""
    grad_x = sobel_filter_x(img)
    grad_y = sobel_filter_y(img)
    # prepatring mix x+y
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.float32)
    return grad


def roberts_filter_x(img):
    """using a Roberts horizontal filter for an image"""
    grad_x = cv2.filter2D(img, cv2.CV_32F, roberts_x)
    return grad_x


def roberts_filter_y(img):
    """using a Roberts vertical filter for an image"""
    grad_y = cv2.filter2D(img, cv2.CV_32F, roberts_y)
    return grad_y


def roberts_filter(img):
    """using a Roberts filter for an image"""
    grad_x = roberts_filter_x(img)
    grad_y = roberts_filter_y(img)
    # prepatring mix x+y
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.float32)
    return grad


def prewitta_filter_x(img):
    """using a Prewitta horizontal filter for an image"""
    grad_x = cv2.filter2D(img, cv2.CV_32F, prewitta_x)
    return grad_x


def prewitta_filter_y(img):
    """using a Prewitta vertical filter for an image"""
    grad_y = cv2.filter2D(img, cv2.CV_32F, prewitta_y)
    return grad_y


def prewitta_filter(img):
    """using a Prewitta filter for an image"""
    grad_x = cv2.filter2D(img, cv2.CV_32F, prewitta_x)
    grad_y = cv2.filter2D(img, cv2.CV_32F, prewitta_y)
    # prepatring mix x+y
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.float32)
    return grad


def laplace_filter(img):
    """using a Laplace 3x3 filter for an image"""
    return cv2.filter2D(img, cv2.CV_32F, laplace)


def median_filter(img, r):
    """using median filter radius r"""
    result = cv2.medianBlur(img, r)
    return result


def maximum_filter(img, r):
    """using maximum filter """
    # cv2.MORPH_RECT is a shape and (r,r) is a kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))
    # using maximum filter with kernel n x n
    result = cv2.dilate(img, kernel)
    return result


def minimum_filter(img, r):
    """using minimum filter """
    # Creates the kernel
    # cv2.MORPH_RECT is a shape and (r,r) is a kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))

    # using the minimum filter with kernel n x n
    result = cv2.erode(img, kernel)
    return result

# let's show them step by step


img = cv2.imread('parrots.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# that's for showing 2 images at same time
plt.figure(2, figsize=(12, 8))
plt.subplot(121)
# that is to show in RGB
plt.imshow(img[..., ::-1])
plt.title("Main Image")
plt.subplot(122)
plt.imshow(img_gray, cmap='gray')
plt.title('Main Image gray scaled(to show next image close this)');
plt.show()

# let's show with sobel filter x, y and mix
img_filtered_x = sobel_filter_x(img_gray)
img_filtered_y = sobel_filter_y(img_gray)
img_filtered = sobel_filter(img_gray)
plt.figure(2, figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_filtered_x, cmap='gray')
plt.title("Horizontal sobel filter")
plt.subplot(122)
plt.imshow(img_filtered_y, cmap='gray')
plt.title('Vertical sobel filter(to show next image close this)');

plt.show()

plt.imshow(img_filtered, cmap='gray')
plt.title('Both Vertical and Horizontal sobel filters(to show next image close this)');
plt.show()

# let's show with roberts filter x, y and mix
img_filtered_x1 = roberts_filter_x(img_gray)
img_filtered_y1 = roberts_filter_y(img_gray)
img_filtered1 = roberts_filter(img_gray)
plt.figure(2, figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_filtered_x1, cmap='gray')
plt.title("Horizontal roberts filter")
plt.subplot(122)
plt.imshow(img_filtered_y1, cmap='gray')
plt.title('Vertical roberts filter(to show next image close this)');

plt.show()
plt.imshow(img_filtered1, cmap='gray')
plt.title('Both Vertical and Horizontal roberts filters(to show next image close this)');
plt.show()

# let's show with prewitta filter x, y and mix
img_filtered_x2 = prewitta_filter_x(img_gray)
img_filtered_y2 = prewitta_filter_y(img_gray)
img_filtered2 = prewitta_filter(img_gray)
plt.figure(2, figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_filtered_x2, cmap='gray')
plt.title("Horizontal prewitta filter")
plt.subplot(122)
plt.imshow(img_filtered_y2, cmap='gray')
plt.title('Vertical prewitta filter(to show next image close this)');
plt.show()

plt.imshow(img_filtered2, cmap='gray')
plt.title('Both Vertical and Horizontal prewitta filters(to show next image close this)');
plt.show()

img_filtered_laplace = laplace_filter(img_gray)
img_filtered_min = minimum_filter(img, 8)
img_filtered_max = maximum_filter(img, 8)
img_filtered_mean = median_filter(img, 11)

plt.figure(2, figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_filtered_laplace,cmap='gray')
plt.title("Laplace filter")
plt.subplot(122)
plt.imshow(img_filtered_min[..., ::-1])
plt.title('Mimimum filter r = 8 (to show next image close this)')
plt.show()


plt.figure(2, figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_filtered_max[..., ::-1])
plt.title("Maximum filter r = 8")
plt.subplot(122)
plt.imshow(img_filtered_mean[..., ::-1])
plt.title('Median filter r = 11 (to show next image close this)')
plt.show()

