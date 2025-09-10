import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time

def main():
    
    print(torch.__version__)
    print(torch.cuda.is_available())

    # 모델 로드
    model = YOLO("yolo11n-seg.pt") #yolov8n.pt "yolo11n.pt"

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
        # print(f"res.probs: {res.probs}")
        
        class_info = []
        index_info = []
        for i, cls in enumerate(res.boxes.cls):
            label = res.names.get(int(cls), "unknown")
            class_info.append(label)
            index_info.append(i)
            print(f"label : {label}, conf : {res.boxes.conf[i]}" )
        
        annotated = results[0].plot()
        
        frames += 1
        fps = (frames / (time.time() - start))
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
       # cell phone의 좌표 얻기
        for label, idx in zip(class_info, index_info):
            if label == 'person':
                mask = res.masks.data[idx].cpu().numpy()             
               
                # frame 에 mask 적용하기 해서 블러 처리하기
                # blurred = cv2.GaussianBlur(frame, (51, 51), 0)
                # frame = np.where(make[:,:,None], blurred, frame)
                
                # frame에 mask 부분만 노란색 색상 칠하기
                # yellow = np.full_like(frame, (0,255,255))
                # frame = np.where(mask[:, :, None], np.clip(frame * 0.7 + yellow * 0.5, 0, 255).astype(np.uint8), frame)
                
                # morphology 처리로 마스크 영상 넓히기
                # kernel = np.ones((15, 15), np.uint8)
                # mask = cv2.dilate(mask.astype(np.uint8), kernel, iteration=5).astype(bool)
                # yellow = np.full_like(frame, (0,255,255))
                # frame = np.where(mask[:, :, None], np.clip(frame * 0.7 + yellow * 0.5, 0, 255).astype(np.uint8), frame)
                
                # morphology 처리고 마스크 영상 좁히기
                # kernel = np.ones((15, 15), np.uint8)
                # mask = cv2.erode(mask.astype(np.uint8), kernel, iteration=2).astype(bool)
                # yellow = np.full_like(frame, (0,255,255))
                # frame = np.where(mask[:, :, None], np.clip(frame * 0.7 + yellow * 0.5, 0, 255).astype(np.uint8), frame)
                
                # morphology 처리로 마스크 영상 열기 (침식 후 팽창)
                # kernel = np.ones((15, 15), np.uint8)
                # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=3).astype(bool)
                # yellow = np.full_like(frame, (0,255,255))
                # frame = np.where(mask[:, :, None], np.clip(frame * 0.7 + yellow * 0.5, 0, 255).astype(np.uint8), frame)
                
                # morphology 처리로 마스크 영상 닫기 (팽창 후 침식)
                kernel = np.ones((15, 15), np.uint8)
                # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=3).astype(bool)
                # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel, iterations=3).astype(bool)
                # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel, iterations=3).astype(bool)
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_HITMISS, kernel, iterations=3).astype(bool)
                # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CROSS, kernel, iterations=3).astype(bool)
                # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_DIAMOND, kernel, iterations=3).astype(bool)
                # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_DILATE, kernel, iterations=3).astype(bool)
                # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_ELLIPSE, kernel, iterations=3).astype(bool)
                # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_ERODE, kernel, iterations=3).astype(bool)
                # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_RECT, kernel, iterations=3).astype(bool)
                    
                #mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_TOPHAT, kernel, iterations=3).astype(bool)

                yellow = np.full_like(frame, (0,255,255))
                frame = np.where(mask[:, :, None], np.clip(frame * 0.7 + yellow * 0.5, 0, 255).astype(np.uint8), frame)
                
                
                
              
        if cv2.waitKey(1) & 0xFF == 27:
            break
        cv2.imshow("YOLOv8 Inference", frame)
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()


