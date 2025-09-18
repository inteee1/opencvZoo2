# pt 파일 경로 설정
# 장치 번호 바꾸기
import cv2
from ultralytics import YOLO

# 1. custom pt 모델 불러오기
# pt 파일 경로 설정
model = YOLO("/home/inteee/opencvZoo2/runs/detect/train6/weights/best.pt")   # 학습한 custom 모델 경로

# 2. 웹캠 열기 (0: 기본 카메라, 숫자를 바꾸면 다른 카메라)
# 장치 번호 바꾸기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
tm = cv2.TickMeter()

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 모델 추론
    results = model(frame, verbose=False)

    # 4. 결과 박스 그리기
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
        confs = r.boxes.conf.cpu().numpy()   # confidence
        clss  = r.boxes.cls.cpu().numpy()    # class index

        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"

            # 사각 박스 (초록색)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 클래스명 + confidence
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # 5. 화면에 출력
    cv2.imshow("YOLOv8 Detection", frame)

    # q 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. 자원 해제
cap.release()
cv2.destroyAllWindows()