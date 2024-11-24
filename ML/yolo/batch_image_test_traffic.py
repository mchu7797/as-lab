import os
import sys
import cv2
from ultralytics import YOLO
from pprint import pprint

# YOLO 모델 로드
model = YOLO('yolo_traffic_light.pt')  # 또는 다른 YOLO 모델 파일 경로

def detect_objects(image_path):
    # 이미지 읽기
    img = cv2.imread(image_path)

    # YOLO로 객체 감지
    results = model.predict(img, imgsz=1280)

    print(f"{image_path}에서 감지된 객체:")
    pprint(results[0].boxes)

    # 결과 시각화
    annotated_img = results[0].plot()

    # 새 이미지 파일 이름 생성
    file_name = os.path.basename(image_path)
    name, ext = os.path.splitext(file_name)
    new_file_name = f"test/{name}-found{ext}"

    # 결과 이미지 저장
    cv2.imwrite(new_file_name, annotated_img)

    print(f"객체 감지 결과가 '{new_file_name}'에 저장되었습니다.")

if __name__ == "__main__":
    # 'test' 폴더 내의 모든 이미지 파일 분석
    test_folder = 'test'
    for file in os.listdir(test_folder):
        if file.endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 확장자
            image_path = os.path.join(test_folder, file)
            detect_objects(image_path)
