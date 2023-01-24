from enum import Enum
import math
from statistics import median

import numpy

DIVIDER = False
K = 2.0

CENT_ALGO_INDEX = 1
CENT_ALGO = ["MEAN", "MEDIAN"]


DIST_ALGO_INDEX = 0
DIST_ALGOS = ["MANHATTAN", "EUCLIDEAN"]
POINTS = [
    (0, 2),
    (0, 0),
    (1.5, 0),
    (5, 0),
    (5, 2),
]
CENTROIDS = [
    (6, 6),
    (4, 6),
    (4, 5),
    (2, 6),
]


class Point:
    def __init__(self, x, y, index=""):
        self.x = x
        self.y = y
        self.index = index

    def __repr__(self):
        return f"P{self.index}({self.x}, {self.y})"


def distance(p1: Point, p2: Point):
    if DIST_ALGOS[DIST_ALGO_INDEX] == DIST_ALGOS[0]:  # Manhatten
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)
    elif DIST_ALGOS[DIST_ALGO_INDEX] == DIST_ALGOS[1]:  # Euchlidean
        return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


def distance_matrix(points, centroids):
    first_row = " "*8
    for point in points:
        first_row += f"({point.x},{point.y}) "
    print(first_row)
    for point1 in centroids:
        if DIVIDER:
            print("-"*len(first_row))
        output = f"({point1.x},{point1.y})"
        if len(output) == 5:
            output += " " * 5
        elif len(output) == 6:
            output += " " * 4

        for point2 in points:
            dist = distance(point1, point2)
            if dist <= K:
                output += f"\033[32m{dist}\033[39m"
            else:
                output += f"{dist}"
            space = len(f"({point2.x},{point2.y}) ")-1
            if len(str(dist)) == 1:
                output += " " * space
            elif len(str(dist)) == 2:
                output += " " * (space-1)
        print(output)


def make_clusters(points, centroids):
    clusters = []
    for _ in range(len(centroids)):
        clusters.append([])
    for point in points:
        closest = (float("inf"), -1)
        for i, centroid in enumerate(centroids):
            dist = distance(point, centroid)
            if dist <= 3:
                if dist < closest[0]:
                    closest = (dist, i)
        if closest[-1] >= 0:
            clusters[closest[1]].append(point)
    print("---")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: ({centroids[i].x}, {centroids[i].y})")
        print(cluster)
    return clusters


def find_next_centriod(clusters):
    centroids = []
    for cluster in clusters:
        if CENT_ALGO[CENT_ALGO_INDEX] == CENT_ALGO[0]:
            x = 0
            y = 0
            for point in cluster:
                x += point.x
                y += point.y
            x /= len(cluster)
            y /= len(cluster)
        elif CENT_ALGO[CENT_ALGO_INDEX] == CENT_ALGO[1]:
            x = []
            y = []
            for point in cluster:
                x.append(point.x)
                y.append(point.y)
            x = median(x)
            y = median(y)
        centroids.append(Point(x, y))
    print("---")
    print(centroids)
    return centroids


if __name__ == '__main__':
    points = [Point(node[0], node[1], (i+1)) for i, node in enumerate(POINTS)]
    # centroids = [Point(node[0], node[1]) for node in CENTROIDS]
    # distance_matrix(points, centroids)
    old_cluster = [[points[0], points[1], points[3]], [points[2], points[4]]]
    centroids = find_next_centriod(old_cluster)

    while True:
        clusters = make_clusters(points, centroids)
        compare = numpy.array_equiv(numpy.array(
            old_cluster, dtype=object), numpy.array(clusters, dtype=object))
        if compare:
            print("No change, exiting...")
            break
        old_cluster = clusters.copy()
        centroids = find_next_centriod(clusters)
