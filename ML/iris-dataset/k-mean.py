import csv
import random
from dataclasses import dataclass
from typing import List


@dataclass
class Iris:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    iris_class: str


def preprocess_iris_data(iris_data_path: str) -> List[Iris]:
    with open(iris_data_path, encoding="utf-8") as iris_file:
        iris_dataset = []
        iris_raw_dataset = csv.DictReader(iris_file)

        for row in iris_raw_dataset:
            iris_dataset.append(
                Iris(
                    float(row["sepal.length"]),
                    float(row["sepal.width"]),
                    float(row["petal.length"]),
                    float(row["petal.width"]),
                    row["variety"],
                )
            )

        return iris_dataset


def initialize_centroids(
        n_clusters: int, iris_dataset: List[Iris]
) -> List[List[float]]:
    centroids = []
    iris_data = random.sample(iris_dataset, n_clusters)

    for iris in iris_data:
        centroid = []

        centroid.append(iris.sepal_length)
        centroid.append(iris.sepal_width)
        centroid.append(iris.petal_length)
        centroid.append(iris.petal_width)

        centroids.append(centroid)

    return centroids


def assign_clusters(
        centroids: List[List[float]], iris_dataset: List[Iris]
) -> List[List[Iris]]:
    clusters = [[] for _ in centroids]

    for iris_data in iris_dataset:
        distances = []

        for centroid in centroids:
            distance = 0.0

            distance += (centroid[0] - iris_data.sepal_length) ** 2
            distance += (centroid[1] - iris_data.sepal_width) ** 2
            distance += (centroid[2] - iris_data.petal_length) ** 2
            distance += (centroid[3] - iris_data.petal_width) ** 2

            distances.append(distance)

        clusters[distances.index(min(distances))].append(iris_data)

    return clusters


def update_centroids(
        clusters: List[List[Iris]], old_centroids: List[List[float]]
) -> List[List[float]]:
    new_centroids = []
    learning_rate = 0.1

    for cluster, old_centroid in zip(clusters, old_centroids):
        if not cluster:
            new_centroids.append(old_centroid)
            continue

        means = [0.0 for _ in range(4)]
        for iris_data in cluster:
            means[0] += iris_data.sepal_length
            means[1] += iris_data.sepal_width
            means[2] += iris_data.petal_length
            means[3] += iris_data.petal_width

        means = [mean / len(cluster) for mean in means]

        updated_centroids = [
            old_centroid[i] + learning_rate * (means[i] - old_centroid[i])
            for i in range(4)
        ]

        new_centroids.append(updated_centroids)

    return new_centroids


if __name__ == "__main__":
    iris_dataset = preprocess_iris_data("iris.csv")
    centroids = initialize_centroids(3, iris_dataset)

    for _ in range(500):
        clusters = assign_clusters(centroids, iris_dataset)
        centroids = update_centroids(clusters, centroids)

    for index, centroid in enumerate(centroids):
        print(f"{index + 1}번째 군집 대푯값 : ", centroid)
