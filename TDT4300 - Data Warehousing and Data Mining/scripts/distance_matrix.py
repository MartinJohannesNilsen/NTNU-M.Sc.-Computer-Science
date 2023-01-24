from tabulate import tabulate

points = {
    "a": (3, 15),
    "b": (3, 13),
    "c": (3, 11),
    "d": (3, 8),
    "e": (3, 6),
    "f": (5, 4),
    "g": (5, 12),
    "h": (7, 14),
    "i": (7, 10),
    "j": (7, 6),
    "k": (13, 6),
    "l": (16, 10),
    "m": (13, 13),
}


def distance_function(a, b):
    L = min(abs(a[0]-b[0]), abs(a[1]-b[1]))
    return L


def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a, b))


def calculate(simplified=True, dist=distance_function):
    labels = list(points.keys())
    res = [[""] + labels]
    for i, x in enumerate(points.values()):
        array = [labels[i]]
        for j, y in enumerate(points.values()):
            if simplified and j >= i:
                break
            else:
                array.append(dist(y, x))
        res.append(array)

    print(tabulate(res, headers="firstrow"))


calculate()
