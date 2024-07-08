from math import sqrt

raw_csv_data = []

for line in open("iris.csv", encoding="utf-8"):
    raw_csv_line = [item.strip() for item in line.split(",")]
    raw_csv_data.append(raw_csv_line)

iris_dataset = {}

for csv_row in raw_csv_data:
    try:
        iris_data = {
            "iris_sepal_length": float(csv_row[0]),
            "iris_sepal_width": float(csv_row[1]),
            "iris_petal_length": float(csv_row[2]),
            "iris_petal_width": float(csv_row[3]),
            "iris_class": csv_row[4].strip('"'),
        }

        if iris_data["iris_class"] not in iris_dataset:
            iris_dataset[iris_data["iris_class"]] = []

        iris_dataset[iris_data["iris_class"]].append(iris_data)
    except ValueError:
        # When meet the header row
        continue

iris_status = []

for iris_class, iris_data in iris_dataset.items():
    iris_sepal_length = [data["iris_sepal_length"] for data in iris_data]
    iris_sepal_width = [data["iris_sepal_width"] for data in iris_data]
    iris_petal_length = [data["iris_petal_length"] for data in iris_data]
    iris_petal_width = [data["iris_petal_width"] for data in iris_data]

    iris_status.append(
        {
            "iris_class": iris_class,
            "iris_sepal_length": {
                "mean": sum(iris_sepal_length) / len(iris_sepal_length),
                "variance": sum(
                    [
                        (x - sum(iris_sepal_length) / len(iris_sepal_length)) ** 2
                        for x in iris_sepal_length
                    ]
                )
                / len(iris_sepal_length),
                "max": max(iris_sepal_length),
                "min": min(iris_sepal_length),
            },
            "iris_sepal_width": {
                "mean": sum(iris_sepal_width) / len(iris_sepal_width),
                "variance": sum(
                    [
                        (x - sum(iris_sepal_width) / len(iris_sepal_width)) ** 2
                        for x in iris_sepal_width
                    ]
                )
                / len(iris_sepal_width),
                "max": max(iris_sepal_width),
                "min": min(iris_sepal_width),
            },
            "iris_petal_length": {
                "mean": sum(iris_petal_length) / len(iris_petal_length),
                "variance": sum(
                    [
                        (x - sum(iris_petal_length) / len(iris_petal_length)) ** 2
                        for x in iris_petal_length
                    ]
                )
                / len(iris_petal_length),
                "max": max(iris_petal_length),
                "min": min(iris_petal_length),
            },
            "iris_petal_width": {
                "mean": sum(iris_petal_width) / len(iris_petal_width),
                "variance": sum(
                    [
                        (x - sum(iris_petal_width) / len(iris_petal_width)) ** 2
                        for x in iris_petal_width
                    ]
                )
                / len(iris_petal_width),
                "max": max(iris_petal_width),
                "min": min(iris_petal_width),
            },
        }
    )

print("붓꽃 데이터 통계")

print()
print("------------------------------------------")
print()

for iris_stat in iris_status:
    print("품종:", iris_stat["iris_class"])

    print()

    print("꽃받침 길이 평균:", iris_stat["iris_sepal_length"]["mean"])
    print("꽃받침 길이 분산:", iris_stat["iris_sepal_length"]["variance"])
    print("꽃받침 길이 표준편차:", sqrt(iris_stat["iris_sepal_length"]["variance"]))
    print("꽃받침 길이 최대값:", iris_stat["iris_sepal_length"]["max"])
    print("꽃받침 길이 최소값:", iris_stat["iris_sepal_length"]["min"])

    print()

    print("꽃받침 너비 평균:", iris_stat["iris_sepal_width"]["mean"])
    print("꽃받침 너비 분산:", iris_stat["iris_sepal_width"]["variance"])
    print("꽃받침 너비 표준편차:", sqrt(iris_stat["iris_sepal_width"]["variance"]))
    print("꽃받침 너비 최대값:", iris_stat["iris_sepal_width"]["max"])
    print("꽃받침 너비 최소값:", iris_stat["iris_sepal_width"]["min"])

    print()

    print("꽃잎 길이 평균 : ", iris_stat["iris_petal_length"]["mean"])
    print("꽃잎 길이 분산 : ", iris_stat["iris_petal_length"]["variance"])
    print("꽃잎 길이 표준편차 : ", sqrt(iris_stat["iris_petal_length"]["variance"]))
    print("꽃잎 길이 최대값 : ", iris_stat["iris_petal_length"]["max"])
    print("꽃잎 길이 최소값 : ", iris_stat["iris_petal_length"]["min"])

    print()

    print("꽃잎 너비 평균 : ", iris_stat["iris_petal_width"]["mean"])
    print("꽃잎 너비 분산 : ", iris_stat["iris_petal_width"]["variance"])
    print("꽃잎 너비 표준편차 : ", sqrt(iris_stat["iris_petal_width"]["variance"]))
    print("꽃잎 너비 최대값 : ", iris_stat["iris_petal_width"]["max"])
    print("꽃잎 너비 최소값 : ", iris_stat["iris_petal_width"]["min"])

    print()
    print("------------------------------------------")
    print()
