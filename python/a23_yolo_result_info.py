import cv2
import numpy as np
from ultralytics import YOLO
import torch
#pip install ultralytcs
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import time

def main():
    
    print(torch.__version__)
    print(torch.cuda.is_available())

    # 모델 로드
    model = YOLO("yolo11n.pt") #yolov8n.pt "yolo11n.pt"

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    start = time.time()
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, stream=False, verbose = False)
        res = results[0]
        # print(f"res.boxes: {res.boxes}")
        # print(f"res.names: {res.names}")
        # print(f"res.keypoints: {res.keypoints}")
        # print(f"res.masks: {res.masks}")
        
        class_info = []
        
        for i, cls in enumerate(res.boxes.cls):
            label = res.names.get(int(cls), "unknown")
            class_info.append(label)
            print(f"label : {label}, conf : {res.boxes.conf[i]}" )
        
        # annotated = results[0].plot()
        
        frames += 1
        fps = (frames / (time.time() - start))
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # cell phone의 좌표 얻기
        for label in class_info:
            if label == 'cell phone':
                idx = class_info.index(label)
                bbox = res.boxes.xyxy[idx]
                print(f"cell phone의 좌표 : {bbox}")
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                
                # roi 대상으로 검은색 화면으로 바꾸기
                roi = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                # 검은색 처리
                # roi[::] = 0
                # 블러 처리
                # cv2.blur(roi, (51, 51), roi)
                # canny
                cv2.Canny(roi, 51, 51)
                
              
        if cv2.waitKey(1) & 0xFF == 27:
            break
        cv2.imshow("YOLOv8 Inference", frame)
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()