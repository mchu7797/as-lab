import cv2
import numpy as np
import os
from ultralytics import YOLO


def detect_people(frame, model):
    results = model(frame)
    boxes = []
    for r in results:
        for box in r.boxes:
            if box.cls == 0:  # 0은 보통 'person' 클래스를 나타냅니다
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h] 형식으로 변환
    return np.array(boxes)


def calculate_box_overlap_ratio(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def track_people(prev_boxes, curr_boxes, threshold=0.5):
    tracked_boxes = []
    for prev_box in prev_boxes:
        best_overlap_ratio = 0
        best_match = None
        for curr_box in curr_boxes:
            overlap_radio = calculate_box_overlap_ratio(prev_box, curr_box)
            if overlap_radio > best_overlap_ratio:
                best_overlap_ratio = overlap_radio
                best_match = curr_box

        if best_overlap_ratio > threshold:
            tracked_boxes.append(best_match)

    return tracked_boxes


# YOLO 모델 로드
model = YOLO('yolov8n.pt')  # 또는 다른 YOLO 모델 파일 경로

# 이미지 파일 목록 가져오기
image_dir = './frames/'
output_dir = './outputs/'
os.makedirs(output_dir, exist_ok=True)
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])

# 이전 프레임의 박스 초기화
prev_boxes = []

for image_file in image_files:
    # 이미지 읽기
    frame = cv2.imread(os.path.join(image_dir, image_file))

    # 현재 프레임에서 사람 탐지
    curr_boxes = detect_people(frame, model)

    # 이전 프레임과 현재 프레임의 박스 비교 및 추적
    if len(prev_boxes) > 0:
        tracked_boxes = track_people(prev_boxes, curr_boxes)
    else:
        tracked_boxes = curr_boxes

    # 결과 시각화
    for box in tracked_boxes:
        x, y, w, h = box.astype(int)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 결과 이미지 저장
    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, frame)

    # 현재 박스를 이전 박스로 업데이트
    prev_boxes = tracked_boxes

    # 결과 표시 (선택사항)
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
