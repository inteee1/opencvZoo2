# 설치
# pip install ultralytics
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def _is_valid_quadrilateral(quad_pts):
        """변환된 사각형이 유효한지 검증"""
        if quad_pts is None or len(quad_pts) != 4:
            return False

        # 4개의 점을 numpy 배열로 변환
        pts = quad_pts.reshape(4, 2)

        # 각 변의 길이 계산
        edges = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            edge_length = np.linalg.norm(p2 - p1)
            edges.append(edge_length)

        # 변의 길이 비율 검사 (너무 극단적인 비율은 무효)
        min_edge = min(edges)
        max_edge = max(edges)
        if min_edge == 0 or max_edge / min_edge > 10:
            return False

        # 볼록 사각형인지 확인 (시계 방향 또는 반시계 방향)
        def cross_product_sign(p1, p2, p3):
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

        signs = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            p3 = pts[(i + 2) % 4]
            signs.append(cross_product_sign(p1, p2, p3))

        # 모든 외적의 부호가 같아야 볼록 사각형
        positive = sum(1 for s in signs if s > 0)
        negative = sum(1 for s in signs if s < 0)

        if positive != 4 and negative != 4:
            return False

        return True


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())

    # 모델 로드
    model = YOLO("yolo11s-cls.pt")  # yolo11n-cls.py

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    start = time.time()
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, stream=False, verbose=False)

        res = results[0]
        # print(f"res.boxes: {res.boxes}")
        # print(f"res.names: {res.names}")
        # print(f"res.keypoints: {res.keypoints}")
        # print(f"res.masks: {res.masks}")
        # print(f"res.probes: {res.probs}")

        annotated = results[0].plot()
        frames += 1
        fps = (frames / (time.time() - start))
        # cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # print top1 class
        if res.probs.top1 is not None:
            print(f"Top: {res.probs.top1}, {res.names[res.probs.top1]}")
            print(f"Top1 confidence: {res.probs.top1conf.cpu().numpy()}")

        if cv2.waitKey(1) & 0xFF == 27:
            break
        cv2.imshow("YOLOv8 Inference", annotated)


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
