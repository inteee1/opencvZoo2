import numpy as np
import cv2

def main():
    image = cv2.imread("/home/ab123/opencvZoo2/data/사진 아무거나.jpg", cv2.IMREAD_COLOR)
    logo = cv2.imread("/home/ab123/opencvZoo2/data/아무거나.png", cv2.IMREAD_COLOR)
    if image is None or logo is None: raise Exception("영상파일 읽기 오류")

    cv2.resize(logo, (100, 100), logo)
    #색반전
    logo = cv2.bitwise_not(logo)

    masks = cv2.threshold(logo, 220, 225, cv2.THRESH_BINARY)[1]
    masks = cv2.split(masks)
    #morph 확장
    #masks = [cv2.morphologyEx(m, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations=1)]

    fg_pass_mask = cv2.bitwise_or(masks[0], masks[1])
    fg_pass_mask = cv2.bitwise_or(masks[2], fg_pass_mask)
    bg_pass_mask = cv2.bitwise_not(fg_pass_mask)

    (H, W), (h, w) = image.shape[:2], logo.shape[:2]
    #정 중앙
    x, y = (W - w) // 2, (H - h) // 2
    #x, y = 10, 10
    roi = image[y:y+h, x:x+w] 

    foreground = cv2.bitwise_and(logo, logo, mask=fg_pass_mask)
    background = cv2.bitwise_and(roi, roi, mask=bg_pass_mask)

    dst = cv2.add(foreground, background)
    image[y:y+h, x:x+w] = dst

    cv2.imshow("background", background)
    cv2.imshow("foreground", background)
    cv2.imshow("dst", dst)
    cv2.imshow("image", background)
    cv2.imshow("logo", logo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
if __name__ == "__main__":
    main()