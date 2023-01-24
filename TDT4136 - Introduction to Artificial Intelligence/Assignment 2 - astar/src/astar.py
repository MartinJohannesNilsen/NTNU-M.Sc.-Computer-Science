import math
from heapq import *
from Map import Map_Obj


class node:
    def __init__(self, cords, cost):
        self.cords = cords
        self.cost = cost
        self.est_cost_to_goal = -1
        self.cost_to_node = float("inf")
        self.est_total_cost = -1
        self.expanded = False
        self.best_parent = None
        self.children = []
        self.linked_nodes = []

    # Override comparison method for use in heapq package
    def __lt__(self, other):
        self_smaller = False
        if self.est_total_cost < other.est_total_cost:
            self_smaller = True
        elif self.est_total_cost == other.est_total_cost:
            self_smaller = self.est_cost_to_goal < other.est_cost_to_goal
        return self_smaller


def euclidean(pos, goal):
    x_diff = abs(goal[0] - pos[0])
    y_diff = abs(goal[1] - pos[1])
    return math.sqrt(x_diff**2 + y_diff**2)


def manhattan(pos, goal):
    x_diff = abs(pos[0] - goal[0])
    y_diff = abs(pos[1] - goal[1])
    return x_diff + y_diff


def get_nodes(map, heuristic="m") -> node:
    """Create two-dim array of linked nodes

    Args:
        map (Map_Obj): map object
        heuristic (str, optional): Heuristic, either "m" for manhattan or "e" for euclidean. Defaults to "m".

    Returns:
        node: Two-dimensional array of linked nodes
    """
    nodes = []

    # Read map from file and append to 2dim array
    with open(map.path_to_map, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            row_node_list = []
            for j, val in enumerate([int(_) for _ in line.split(",")]):
                n = n = node([i, j], val)
                n.est_cost_to_goal = manhattan(
                    n.cords, map.goal_pos) if heuristic == "m" else euclidean(n.cords, map.goal_pos)
                row_node_list.append(n)
            nodes.append(row_node_list)

    # Update links for each node, connecting all the nodes with edges between them
    for i, row in enumerate(nodes):
        for j, n in enumerate(row):
            if n.cost != -1:
                if i > 0:
                    if nodes[i - 1][j].cost != -1:
                        n.linked_nodes.append(nodes[i - 1][j])
                if i < len(nodes) - 1:
                    if nodes[i + 1][j].cost != -1:
                        n.linked_nodes.append(nodes[i + 1][j])
                if j > 0:
                    if nodes[i][j - 1].cost != -1:
                        n.linked_nodes.append(nodes[i][j - 1])
                if j < len(row) - 1:
                    if nodes[i][j + 1].cost != -1:
                        n.linked_nodes.append(nodes[i][j + 1])
    return nodes


if __name__ == "__main__":
    # User input
    task_number = int(input("Task: "))
    heuristic = input("Heuristic [m/e] (manhattan/euclidean): ")
    while heuristic[0].lower() != "m" and heuristic[0].lower() != "e":
        print(heuristic[0])
        heuristic = input("Heuristic [m/e] (manhattan/euclidean): ")
    gif = input("Save gif [y/n]: ")
    while gif[0].lower() != "y" and gif[0].lower() != "n":
        gif = input("Save gif [y/n]: ")
    png = input("Save png [y/n]: ")
    while png[0].lower() != "y" and png[0].lower() != "n":
        png = input("Save png [y/n]: ")

    # Initialize
    map = Map_Obj(task=task_number)
    nodes = get_nodes(map, heuristic=heuristic[0])
    open_nodes = []
    curr_node = nodes[map.start_pos[0]][map.start_pos[1]]
    curr_node.expanded = True
    curr_node.cost_to_node = 0
    heappush(open_nodes, curr_node)

    # Find shortes path to goal
    while curr_node.cords != map.goal_pos:
        curr_node = heappop(open_nodes)
        curr_node.expanded = True

        # Check all nodes in linked_nodes
        for n in curr_node.linked_nodes:
            if not n.expanded:
                cost_to_node = curr_node.cost_to_node + n.cost
                if cost_to_node < n.cost_to_node:
                    n.cost_to_node = cost_to_node
                    n.est_total_cost = n.est_cost_to_goal + cost_to_node
                    n.best_parent = curr_node
                    heappush(open_nodes, n)
        # Visualize the path as gif if requested
        if gif[0].lower() == "y":
            map.append_step_visually(curr_node.cords, gif=True)
    # Create gif if requested by user
    if gif[0].lower() == "y":
        map.create_gif(
            f"Task{task_number}_{'manhattan' if (heuristic == 'm') else 'euclidean'}")

    # List the shortest path by using best parent from goal and back to start
    best_path = list()
    n = curr_node
    while n.cords != map.start_pos:
        best_path.append((n.best_parent.cords[0], n.best_parent.cords[1]))
        n = n.best_parent
    print("Best path:\n", best_path)
    print("Number of steps:", len(best_path))

    print("Visualizing path ...")
    for path in best_path:
        map.append_step_visually(path, gif=False, selected_path=True)
    map.show_map()
    if png[0].lower() == "y":
        map.create_png(
            f"Task{task_number}_{'manhattan' if (heuristic == 'm') else 'euclidean'}")
