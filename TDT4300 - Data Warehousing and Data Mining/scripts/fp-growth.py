from itertools import combinations
from pprint import pprint
import sys

SPLIT = True
transaction_list = [
    ["ACD"],
    ["BCE"],
    ["ABCE"],
    ["BE"],
    ["ABCE"],
    ["ABCD"],
]


def split_items(transaction_list):
    for i, transaction in enumerate(transaction_list):
        transaction_list[i] = list(transaction[0])
    return transaction_list


def unique_items(transaction_set):
    return sorted(list(set(item for items in transaction_set for item in items)))


def n_itemset(items, n):
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


def sort_transactions(item_support):
    new_list = []
    for _ in range(len(transaction_list)):
        new_list.append([])
    for i, transaction in enumerate(transaction_list):
        for key, _ in item_support:
            if key[0] in transaction:
                new_list[i].append(key)
    return new_list


class Node:
    def __init__(self, letter):
        self.letter = letter
        self.value = 0
        self.next = []

    def __repr__(self):
        return f"Node({self.letter}, {self.value}, {self.next})"


def make_tree(t_list):
    root = Node("root")
    for t in t_list[:]:
        next_node = root
        for item in t:
            item_exists = False
            for n in next_node.next:
                if(n.letter == item[0]):
                    n.value += 1
                    next_node = n
                    item_exists = True
                    break
            if not item_exists:
                n = Node(item[0])
                n.value += 1
                next_node.next.append(n)
                next_node = n
    return root


def print_tree(root, string, indent):
    for node in root.next:
        string += "\n"
        string += " "*indent
        string += f"|--{node.letter}({node.value})"
        string = print_tree(node, string, indent+3)
    return string


if __name__ == '__main__':
    try:
        support = float(sys.argv[1])
    except:
        print("Husk å sende med verdier support")
        exit()
    print(f"{support=}")
    if SPLIT:
        transaction_list = split_items(transaction_list)
    items = unique_items(transaction_list)
    n_itemset = n_itemset(items, 1)
    freq_set = frequency(n_itemset)
    item_frequency = sorted(freq_set.items(), key=lambda x: x[1], reverse=True)
    print("item_frequency:")
    pprint(item_frequency)
    freq_set = fiter_support(freq_set, support)
    item_frequency = sorted(freq_set.items(), key=lambda x: x[1], reverse=True)
    sorted_transaction_list = sort_transactions(item_frequency)
    print("sorted_transaction_list:")
    pprint(sorted_transaction_list)
    root = make_tree(sorted_transaction_list)
    tree = print_tree(root, f"{root.letter}", 0)
    print("tree:")
    print(tree)
