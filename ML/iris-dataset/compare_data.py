import csv
from collections import defaultdict
from statistics import mean, variance
from math import sqrt
from itertools import combinations

# 전역 상수로 영어-한국어 변환 딕셔너리 정의
FEATURE_NAMES = {
    'sepal_length': '꽃받침 길이',
    'sepal_width': '꽃받침 너비',
    'petal_length': '꽃잎 길이',
    'petal_width': '꽃잎 너비'
}

CSV_TO_FEATURE_NAMES = {
    'sepal.length': 'sepal_length',
    'sepal.width': 'sepal_width',
    'petal.length': 'petal_length',
    'petal.width': 'petal_width'
}

def load_iris_data(filename):
    iris_dataset = defaultdict(list)
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            iris_class = row['variety']
            iris_data = {
                feature_name: float(row[csv_name])
                for csv_name, feature_name in CSV_TO_FEATURE_NAMES.items()
            }
            iris_dataset[iris_class].append(iris_data)
    return iris_dataset

def calculate_statistics(data_list):
    values = [item for item in data_list if item is not None]
    return {
        'mean': mean(values),
        'variance': variance(values),
        'std_dev': sqrt(variance(values)),
        'max': max(values),
        'min': min(values)
    }

def analyze_iris_data(iris_dataset):
    iris_stats = {}
    for iris_class, iris_data in iris_dataset.items():
        class_stats = {}
        for feature in FEATURE_NAMES.keys():
            feature_values = [data[feature] for data in iris_data][:30]
            class_stats[feature] = calculate_statistics(feature_values)
        iris_stats[iris_class] = class_stats
    return iris_stats

def iris_compare(iris_stats, iris_data, compare_option):
    distance_value = {}
    for iris_class, iris_stat in iris_stats.items():
        distance_value[iris_class] = 0
        for compare_feature in compare_option:
            distance_value[iris_class] += (iris_stat[compare_feature]["mean"] - iris_data[compare_feature]) ** 2
    return min(distance_value.items(), key=lambda x: x[1])

def calculate_misclassifications(iris_dataset, iris_stats, compare_option):
    error_count = 0
    for true_class, class_data in iris_dataset.items():
        for i in range(30, 50):
            iris_data = class_data[i]
            predicted_class, _ = iris_compare(iris_stats, iris_data, compare_option)
            if predicted_class != true_class:
                error_count += 1
    return error_count

def main():
    iris_dataset = load_iris_data("iris.csv")
    iris_stats = analyze_iris_data(iris_dataset)
    
    all_features = list(FEATURE_NAMES.keys())
    
    for num_features in [1, 2, 4]:
        print(f"\n{num_features}개 속성 선택 시 결과:")
        for compare_option in combinations(all_features, num_features):
            error_count = calculate_misclassifications(iris_dataset, iris_stats, compare_option)
            print(f"선택된 속성: {', '.join(compare_option)}")
            print(f"오분류 개수: {error_count}")
            print()

if __name__ == "__main__":
    main()