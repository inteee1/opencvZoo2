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
    model = YOLO("yolo11n-obb.pt")  # 배, 비행기 확인

    # cap = cv2.VideoCapture(0)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    img = cv2.imread("/home/inteee/opencvZoo2/data/plane.jpg")
   

    results = model.predict(img, stream=False, verbose=False)

    res = results[0]
    print(f"res.boxes: {res.boxes}")
    print(f"res.names: {res.names}")
    print(f"res.keypoints: {res.keypoints}")
    print(f"res.masks: {res.masks}")
    print(f"res.probes: {res.probs}")
    print(f"res.obb : {res.obb}")

    annotated = results[0].plot()
    # cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # print top1 class
    
   
    cv2.imshow("YOLOv8 Inference", annotated)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
