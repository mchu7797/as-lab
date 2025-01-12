import os
import pandas as pd
from pathlib import Path


def create_labels_csv_balanced(labels_dir, output_csv_path):
    """
    YOLO 형식의 라벨 파일들을 읽어서,
    1) 가장 적은 이미지 수를 가진 클래스를 기준으로 샘플 수를 통일하고
    2) 처음 나온 클래스 순서대로 라벨 번호를 매겨
    최종 CSV 파일을 생성합니다.

    Args:
        labels_dir: 라벨 txt 파일들이 있는 디렉토리 경로
        output_csv_path: 생성할 CSV 파일 경로
    """
    data_list = []
    classes_in_appearance_order = []  # 처음 등장하는 순서를 기록할 리스트

    # labels 디렉토리 내의 모든 txt 파일 처리
    for txt_file in os.listdir(labels_dir):
        if txt_file.endswith('.txt'):
            # 이미지 파일명 구하기 (확장자만 다름)
            base_name = os.path.splitext(txt_file)[0]
            img_file = base_name + '.jpg'  # 또는 .png, .jpeg 등

            # txt 파일에서 클래스 정보 읽기 (첫 번째 bounding box만 사용)
            with open(os.path.join(labels_dir, txt_file), 'r') as f:
                line = f.readline().strip()
                if not line:
                    continue
                class_id = int(line.split()[0])  # 첫 번째 숫자가 클래스 ID라고 가정

            # 처음 등장하는 클래스는 순서 리스트에 추가
            if class_id not in classes_in_appearance_order:
                classes_in_appearance_order.append(class_id)

            # 데이터를 임시로 저장(원본 클래스 ID 포함)
            data_list.append({
                'image_filename': img_file,
                'original_class_id': class_id
            })

    # DataFrame 생성
    df = pd.DataFrame(data_list)

    # (1) 클래스 순서대로 매핑(dict) 만들기
    # 예: classes_in_appearance_order = [7, 10, 2] -> {7: 0, 10: 1, 2: 2}
    class_mapping = {c: idx for idx, c in enumerate(classes_in_appearance_order)}

    # (2) 매핑을 이용해 vehicle_type 칼럼 생성
    # 예: 'class_0', 'class_1', 'class_2' 등의 형태로 라벨링
    df['vehicle_type'] = df['original_class_id'].map(lambda x: f'class_{class_mapping[x]}')

    # 클래스별 샘플 수 확인
    class_counts = df['vehicle_type'].value_counts()
    print("원본 클래스 분포:")
    print(class_counts, "\n")

    # 최소 클래스 수 파악
    min_count = class_counts.min()
    print(f"가장 적은 이미지 수 (min_count) = {min_count}")

    # 각 클래스별로 min_count만큼 샘플링하여 균형 맞추기
    df_balanced = (
        df.groupby('vehicle_type', group_keys=False)
        .apply(lambda x: x.sample(min_count, random_state=42))  # 필요 시 random_state 제거 가능
        .reset_index(drop=True)
    )

    # CSV 파일 생성
    df_balanced.to_csv(output_csv_path, index=False)

    print(f"\n[Balanced] CSV 파일 생성 완료: {len(df_balanced)}개의 데이터가 처리되었습니다.")
    print(f"저장 위치: {output_csv_path}")
    print("\n샘플링 이후 클래스 분포:")
    print(df_balanced['vehicle_type'].value_counts())


# 사용 예시
if __name__ == '__main__':
    labels_dir = 'D:\\AS-Lab\\ML\\CNN\\car_train\\labels'  # 라벨 파일이 있는 디렉토리
    output_csv = 'D:\\AS-Lab\\ML\\CNN\\car_train\\labels.csv'  # 생성할 CSV 파일 경로
    create_labels_csv_balanced(labels_dir, output_csv)
