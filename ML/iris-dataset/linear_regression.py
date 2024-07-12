import math

def mean(values):
    return sum(values) / len(values)

def linear_regression(data):
    # 데이터에서 x와 y 분리
    x = [point[0] for point in data]
    y = [point[1] for point in data]

    # x와 y의 평균 계산
    x_mean = mean(x)
    y_mean = mean(y)

    # 기울기(m)와 절편(b) 계산을 위한 값들
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    # 기울기(m) 계산
    m = numerator / denominator

    # 절편(b) 계산
    b = y_mean - m * x_mean

    return m, b

# 예시 데이터
example_data = [
    (50, 47),
    (37, 40),
    (45, 50),
    (20, 60),
    (31, 50)
]

# 선형 회귀 수행
slope, intercept = linear_regression(example_data)

print(f"1차 함수: y = {slope:.2f}x + {intercept:.2f}")
print("40을 넣었을때에는? :", slope * 40 + intercept)
