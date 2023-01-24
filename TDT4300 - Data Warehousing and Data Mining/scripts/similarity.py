import numpy as np


def cosine(x: np.array, y: np.array, calculation: False):
    numerator = x@y
    l2norm_x = np.linalg.norm(x)
    l2norm_y = np.linalg.norm(y)
    denominator = l2norm_x*l2norm_y
    res = numerator/denominator
    if calculation:
        print("-"*10, f"Calculation", "-"*10)
        print("Cos(x,y) = x*y / ||x|| * ||y||")
        string = "  x*y = "
        for i, (a, b) in enumerate(zip(x, y)):
            if i != 0:
                string += " + "
            string += f"{a}*{b}"
        print(f"{string} = {numerator}")
        string = "  ||x|| = √("
        for i, a in enumerate(x):
            if i != 0:
                string += " + "
            string += f"{a}^2"
        print(f"{string}) = {l2norm_x}")
        string = "  ||y|| = √("
        for i, b in enumerate(y):
            if i != 0:
                string += " + "
            string += f"{b}^2"
        print(f"{string}) = {l2norm_y}")
        print(f"Cos(x,y) = {numerator}/({l2norm_x}*{l2norm_y}) = {res}\n")
        print("-"*10, f"Result", "-"*10)
    return round(res, 3)


def correlation(x: np.array, y: np.array, calculation: False):
    """Using Pearson's product moment correlation coefficient"""
    # from scipy import stats
    # res, p = stats.pearsonr(x, y)
    mx = round(np.mean(x), 3)
    my = round(np.mean(y), 3)
    if calculation:
        print("-"*10, f"Calculation", "-"*10)
        print("Corr_coff(x,y) = ∑(x-mx)(y-my) / √(∑(x-mx)^2 * ∑(y-my)^2)")
        print(f"  mx (mean of x) = {round(mx, 2)}")
        print(f"  my (mean of y) = {round(my, 2)}")
        string = "  (1) ∑(x-mx)(y-my) = "
        sum_1 = 0
        for i, (a, b) in enumerate(zip(x, y)):
            if i != 0:
                string += " + "
            string += f"({a}-{mx})*({b}-{my})"
            sum_1 += (a-mx)*(b-my)
        print(f"{string} = {sum_1}")
        string = "  (2) ∑(x-mx)^2 = "
        sum_2 = 0
        for i, a in enumerate(x):
            if i != 0:
                string += " + "
            string += f"({a}-{mx})^2"
            sum_2 += (a-mx)**2
        print(f"{string} = {sum_2}")
        string = "  (3) ∑(y-my)^2 = "
        sum_3 = 0
        for i, a in enumerate(y):
            if i != 0:
                string += " + "
            string += f"({a}-{mx})^2"
            sum_3 += (a-mx)**2
        print(f"{string} = {sum_2}")
        res = sum_1/np.sqrt(sum_2*sum_3)
        print(
            f"Corr_coff(x,y) = (1)/√(2)*(3) = {sum_1}/√({sum_2}*{sum_3}) = {res}\n")
        print("-"*10, f"Result", "-"*10)
    return round(res, 3)


def jaccard(x: np.array, y: np.array, calculation: False):
    numerator = np.minimum(x, y)
    denominator = np.maximum(x, y)
    res = np.sum(numerator)/np.sum(denominator)
    if calculation:
        print("-"*10, f"Calculation", "-"*10)
        print("Jaccard(x,y) = ∑min(x,y) / ∑max(x,y)")
        string = f" = {numerator}/{denominator}"
        string_numerator = ""
        string_denominator = ""
        for i, (a, b) in enumerate(zip(numerator, denominator)):
            if i != 0:
                string_numerator += "+"
                string_denominator += "+"
            string_numerator += f"{a}"
            string_denominator += f"{b}"
        print(f"{string}\n = ({string_numerator}) / ({string_denominator})\n = {res}\n")

        print("-"*10, f"Result", "-"*10)
    return round(res, 3)


def euclidean(x: np.array, y: np.array, calculation: False):
    res = np.linalg.norm(x-y)
    if calculation:
        print("-"*10, f"Calculation", "-"*10)
        print("Euclidean(x,y) = √∑(p-q)^2)")
        string = " = "
        for i, (a, b) in enumerate(zip(x, y)):
            if i != 0:
                string += " + "
            string += f"({a}-{b})^2"
        print(f"{string} \n = {res}\n")
        print("-"*10, f"Result", "-"*10)
    return round(res, 3)


if __name__ == "__main__":

    CALCULATION = True
    x = np.array([0, 1, 1, 1])
    y = np.array([1, 1, 1, 0])

    print("%"*30, "Cosine Similarity", "%"*30, sep="\n", end="\n\n")
    print(cosine(x, y, CALCULATION))
    print("\n", "%"*30,
          "Pearson's product moment correlation coefficient", "%"*30, sep="\n", end="\n\n")
    print(correlation(x, y, CALCULATION))
    print("\n", "%"*30, "Jaccard Similarity", "%"*30, sep="\n", end="\n\n")
    print(jaccard(x, y, CALCULATION))
    print("\n", "%"*30, "Euclidean distance", "%"*30, sep="\n", end="\n\n")
    print(euclidean(x, y, CALCULATION))
