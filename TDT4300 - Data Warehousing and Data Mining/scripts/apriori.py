from itertools import combinations
from pprint import pprint
import sys
SPLIT = True
transaction_list = [
    ["ACDEF"],
    ["ABCDE"],
    ["BCF"],
    ["ACDEF"],
    ["DB"],
]


def split_items(transaction_list):
    for i, transaction in enumerate(transaction_list):
        transaction_list[i] = list(transaction[0])
    return transaction_list


def unique_items(transaction_list):

    return sorted(list(set(item for items in transaction_list for item in items)))


def generate_n_itemset(items, n):
    return list(combinations(items, n))


def frequency(freq_list):
    freq_set = dict((el, 0) for el in freq_list)
    for key, _ in freq_set.items():
        for transaction in transaction_list:
            check = all(item in transaction for item in key)
            if check:
                freq_set[key] += 1
    return freq_set


def fiter_support(freq_set, support=0.5):
    return {k: v for k, v in freq_set.items() if v >= (len(transaction_list) * support)}


# if __name__ == '__main__':
#     try:
#         n = int(sys.argv[1])
#         support = float(sys.argv[2])
#     except:
#         print("Husk å sende med verdier n og support")
#         exit()
#     print(f"{n=}")
#     if SPLIT:
#         transaction_list = split_items(TRANSACTION_LIST)
#     else:
#         transaction_list = TRANSACTION_LIST
#     items = unique_items(transaction_list)

#     n_itemset = n_itemset(items, n)
#     freq_set = frequency(n_itemset)
#     pprint(freq_set)
if __name__ == '__main__':
    try:
        support = float(sys.argv[1])
    except:
        print("Husk å sende med support")
        exit()
    if SPLIT:
        transaction_list = split_items(transaction_list)
    n = 1
    items = unique_items(transaction_list)
    while 1:
        n_itemset = generate_n_itemset(items, n)
        freq_set = frequency(n_itemset)
        if not len(freq_set.items()):
            print("Funnet siste elementset!")
            break
        pprint(freq_set)
        freq_set = fiter_support(freq_set, support)
        # pprint(freq_set)
        items = unique_items([i for k, _ in freq_set.items() for i in k])
        n += 1
