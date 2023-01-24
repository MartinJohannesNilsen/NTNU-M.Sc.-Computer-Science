import numpy as np


def print_each():
    for x in range(32, 300, 64):
        for y in range(32, 300, 64):
            print(f"({x},{y})")
            input("done?")


def print_anchor_box_sizes_handout():
    aspect_ratios = [2, 3]
    min_size = 162
    next_min_size = 213
    for ar in aspect_ratios:
        print(f"Aspect ratio {ar}")
        print("Handout functions:")
        print(min_size**2)
        print(round(min_size*next_min_size))
        print(f"[{round(min_size*np.sqrt(ar))}, {round(min_size/np.sqrt(ar))}]")
        print(f"[{round(min_size/np.sqrt(ar))}, {round(min_size*np.sqrt(ar))}]")


def print_anchor_box_sizes_blogpost():
    aspect_ratios = [2, 3]
    min_size = 162
    next_min_size = 213
    for ar in aspect_ratios:
        print(f"Aspect ratio {ar}")
        print("Blogpost functions:")
        scale = np.sqrt(min_size*next_min_size)
        w = scale * np.sqrt(ar)
        h = scale / np.sqrt(ar)
        w, h = round(w), round(h)
        print(f"[{w}, {h}]")
        print(f"[{h}, {w}]")


if __name__ == "__main__":
    # print_each()
    print_anchor_box_sizes_handout()
    # print_anchor_box_sizes_blogpost()
