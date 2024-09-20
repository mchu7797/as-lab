import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("yolov8n.pt")

# 웹캠 초기화
cap = cv2.VideoCapture(0)

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델로 객체 감지
    results = model(frame)

    # 결과 시각화
    annotated_frame = results[0].plot()

    # 처리된 프레임 표시
    cv2.imshow("YOLO 객체 감지", annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 자원 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
