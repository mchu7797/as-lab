import csv
from collections import defaultdict
from typing import Dict, List, Tuple

# 전역 상수
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

def load_iris_data(filename: str) -> Dict[str, List[Dict[str, float]]]:
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

def get_mean(values: List[float]) -> float:
    return sum(values) / len(values)

def extract_feature_data(iris_data: List[Dict[str, float]], feature: str) -> List[float]:
    return [data[feature] for data in iris_data]

def learn_iris_data(iris_data: List[Dict[str, float]], start: int, end: int) -> Tuple[float, float]:
    x_pos = extract_feature_data(iris_data[start:end], 'petal_length')
    y_pos = extract_feature_data(iris_data[start:end], 'petal_width')
    
    x_mean, y_mean = get_mean(x_pos), get_mean(y_pos)
    
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x_pos, y_pos))
    denominator = sum((xi - x_mean) ** 2 for xi in x_pos)
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    return slope, intercept

def calc_error_sum(slope: float, intercept: float, iris_data: List[Dict[str, float]], start: int, end: int) -> float:
    x_pos = extract_feature_data(iris_data[start:end], 'petal_length')
    y_pos = extract_feature_data(iris_data[start:end], 'petal_width')
    
    return sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x_pos, y_pos))

def analyze_iris_data(iris_dataset: Dict[str, List[Dict[str, float]]]) -> Dict[str, float]:
    error_sum = {}
    print("\n선형회귀분석 테스트\n")
    
    for iris_class, iris_data in iris_dataset.items():
        print("-" * 50, "\n")
        slope, intercept = learn_iris_data(iris_data, 0, 30)
        error_sum_by_class = calc_error_sum(slope, intercept, iris_data, 30, 50)
        error_sum[iris_class] = error_sum_by_class
        
        print(f"{iris_class}의 1차 함수: y = {slope:.2f}x + {intercept:.2f}")
        print(f"{iris_class}의 오차 합: {error_sum_by_class:.2f}", "\n")
    
    return error_sum

def main():
    iris_dataset = load_iris_data("iris.csv")
    error_sum = analyze_iris_data(iris_dataset)
    
    print('-' * 50)
    print(f"\n최소 오차 합인 품종 : {min(error_sum, key=error_sum.get)}\n")

if __name__ == "__main__":
    main()