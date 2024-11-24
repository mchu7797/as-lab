import os
import pandas as pd
from pathlib import Path

def create_labels_csv(labels_dir, output_csv_path):
    """
    YOLO 형식의 라벨 파일들을 읽어서 CSV 파일 생성
    
    Args:
        labels_dir: 라벨 txt 파일들이 있는 디렉토리 경로
        output_csv_path: 생성할 CSV 파일 경로
    """
    data_list = []
    
    # labels 디렉토리 내의 모든 txt 파일 처리
    for txt_file in os.listdir(labels_dir):
        if txt_file.endswith('.txt'):
            # 이미지 파일명 구하기 (확장자만 다름)
            base_name = os.path.splitext(txt_file)[0]
            img_file = base_name + '.jpg'  # 또는 다른 확장자를 사용하시면 됩니다
            
            # txt 파일에서 클래스 정보 읽기
            with open(os.path.join(labels_dir, txt_file), 'r') as f:
                class_id = int(f.readline().split()[0])  # 첫 번째 숫자가 클래스 ID
            
            # 데이터 리스트에 추가
            data_list.append({
                'image_filename': img_file,
                'vehicle_type': f'class_{class_id}'  # 클래스 ID를 임시 이름으로 사용
            })
    
    # CSV 파일 생성
    df = pd.DataFrame(data_list)
    df.to_csv(output_csv_path, index=False)
    
    print(f"CSV 파일 생성 완료: {len(data_list)}개의 데이터가 처리되었습니다.")
    print(f"저장 위치: {output_csv_path}")
    print("\n클래스 분포:")
    print(df['vehicle_type'].value_counts())

# 사용 예시
if __name__ == '__main__':
    labels_dir = 'P:\Development\AS-Lab\ML\CNN\car_train\labels'  # 라벨 파일이 있는 디렉토리
    output_csv = 'P:\Development\AS-Lab\ML\CNN\car_train\labels.csv'      # 생성할 CSV 파일 경로
    create_labels_csv(labels_dir, output_csv) 