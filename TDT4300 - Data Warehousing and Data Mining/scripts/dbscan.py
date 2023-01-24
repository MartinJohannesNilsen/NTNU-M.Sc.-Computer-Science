from enum import Enum
import math
from pprint import pprint


MINPTS = 2.0
EPS = 2.0
DIST_ALGO_INDEX = 2
DIST_ALGOS = ["MANHATTAN", "EUCLIDEAN", "L_MIN"]
POINTS = [
    (3, 15),
    (3, 13),
    (3, 11),
    (3, 8),
    (3, 6),
    (5, 4),
    (5, 12),
    (7, 14),
    (7, 10),
    (7, 6),
    (13, 6),
    (13, 13),
    (16, 10),
]


class PointType(Enum):
    CORE = "core"
    BORDER = "border"
    NOISE = "noise"
    UNKNOWN = "unknown"


class Point:
    def __init__(self, x, y, index):
        self.index = index
        self.x = x
        self.y = y
        self.type = PointType.UNKNOWN

    def __repr__(self):
        return f"P{self.index}({self.x}, {self.y}, {self.type})"


def distance(p1: Point, p2: Point):
    if DIST_ALGOS[DIST_ALGO_INDEX] == DIST_ALGOS[0]:  # Manhatten
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)
    elif DIST_ALGOS[DIST_ALGO_INDEX] == DIST_ALGOS[1]:  # Euchlidean
        return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
    elif DIST_ALGOS[DIST_ALGO_INDEX] == DIST_ALGOS[2]:
        return min(abs(p1.x - p2.x), abs(p1.y - p2.y))


def findCore(points):
    for point in points:
        nr_of_neighbours = 0
        is_core_neighbour = False
        for neighbour in points:
            if distance(point, neighbour) <= EPS:
                nr_of_neighbours += 1
                if not is_core_neighbour and neighbour.type == PointType.CORE:
                    is_core_neighbour = True
        if nr_of_neighbours >= MINPTS:
            point.type = PointType.CORE
        elif is_core_neighbour:
            point.type = PointType.BORDER
    return points


def findBorderOrNoise(points):
    corePoints = [point for point in points if point.type == PointType.CORE]
    for point in points:
        if point.type != PointType.UNKNOWN:
            continue

        for core in corePoints:
            if distance(point, core) <= EPS:
                point.type = PointType.BORDER
                break
        if point.type == PointType.UNKNOWN:
            point.type = PointType.NOISE
    return points


def makeClusters(points):
    clusters = []
    corePoints = [point for point in points if point.type == PointType.CORE]
    restPoints = [point for point in points if point not in corePoints]
    while len(corePoints):
        cluster = []
        cluster.append(corePoints.pop(0))
        for point in cluster:
            for i, core in reversed(list(enumerate(corePoints))):
                if distance(point, core) <= EPS:
                    cluster.append(corePoints.pop(i))

        for point in cluster:
            for i, rest in reversed(list(enumerate(restPoints))):
                if distance(point, rest) <= EPS:
                    cluster.append(restPoints.pop(i))

        clusters.append(cluster)

    return clusters


if __name__ == '__main__':
    points = [Point(node[0], node[1], (i+1)) for i, node in enumerate(POINTS)]
    points = findCore(points)
    points = findBorderOrNoise(points)
    nonNoisePoints = [
        point for point in points if point.type != PointType.NOISE]
    clusters = makeClusters(nonNoisePoints)
    print("Points:")
    pprint(points)
    print("Clusters:")
    pprint(clusters)
