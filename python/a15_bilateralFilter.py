import cv2
import numpy as np

def main():
    image = cv2.imread("/home/ab123/opencvZoo2/data/lenna.bmp")
    simple_blurred = cv2.blur(image, (15, 15))
    blurred = cv2.bilateralFilter(image, d=15, sigmaColor=75, sigmaSpace=75)
    gaussian_blurred = cv2.GaussianBlur(image, (15,15), 0)

    cv2.imshow("simple_blurred", simple_blurred)
    cv2.imshow("gaussianblur", gaussian_blurred)
    cv2.imshow("image", image)
    cv2.imshow("blurred", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()