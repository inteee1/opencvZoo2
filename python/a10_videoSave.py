import cv2
import numpy as np


def put_string(frame, text, pt, value, color=(120, 200, 90)):
    text += str(value)
    shade = (pt[0] + 2, pt[1] + 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, shade, font, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, text, pt, font, 0.7, color, 2)


def main():
    cap = cv2.VideoCapture("/home/ab123/opencvZoo2/data/vtest.avi", cv2.IMREAD_GRAYSCALE)

    print(f"너비 {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"높이 {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"포맷 {cap.get(cv2.CAP_PROP_FORMAT)}")
    print(f"노출 {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"밝기 {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter("/home/ab123/opencvZoo2/data/vtest_output.avi", fourcc, fps, size)


    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽지 못 했습니다.")
            break

        put_string(frame, "width", (10, 30), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        put_string(frame, "Height", (10, 60), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        put_string(frame, "fps", (10, 90), cap.get(cv2.CAP_PROP_FPS))

        cv2.imshow("webcam", frame)
        writer.write(frame)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()