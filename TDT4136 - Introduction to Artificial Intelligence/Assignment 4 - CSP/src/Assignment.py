import copy
import itertools


class CSP:
    def __init__(self):
        # self.variables is a list of the variable names in the CSP
        self.variables = []

        # self.domains[X_i] is a list of legal values for variable X_i
        self.domains = {}

        # self.constraints[X_i][X_j] is a list of legal value pairs for
        # the variable pair (X_i, X_j)
        self.constraints = {}

        # variables for storing amount of backtracks and failures
        self.n_backtracks = 0
        self.n_failures = 0

    def add_variable(self, name, domain):
        """Add a new variable to the CSP. 'name' is the variable name
        and 'domain' is a list of the legal values for the variable.
        """
        self.variables.append(name)
        self.domains[name] = list(domain)
        self.constraints[name] = {}

    def get_all_possible_pairs(self, a, b):
        """Get a list of all possible pairs (as tuples) of the values in
        the lists 'a' and 'b', where the first component comes from list
        'a' and the second component comes from list 'b'.
        """
        return itertools.product(a, b)

    def get_all_arcs(self):
        """Get a list of all arcs/constraints that have been defined in
        the CSP. The arcs/constraints are represented as tuples (X_i, X_j),
        indicating a constraint between variable 'X_i' and 'X_j'.
        """
        return [(X_i, X_j) for X_i in self.constraints for X_j in self.constraints[X_i]]

    def get_all_neighboring_arcs(self, var):
        """Get a list of all arcs/constraints going to/from variable
        'var'. The arcs/constraints are represented as in get_all_arcs().
        """
        return [(X_i, var) for X_i in self.constraints[var]]

    def add_constraint_one_way(self, X_i, X_j, filter_function):
        """Add a new constraint between variables 'X_i' and 'X_j'. The legal
        values are specified by supplying a function 'filter_function',
        that returns True for legal value pairs and False for illegal
        value pairs. This function only adds the constraint one way,
        from X_i -> X_j. You must ensure that the function also gets called
        to add the constraint the other way, X_j -> X_i, as all constraints
        are supposed to be two-way connections!
        """
        if not X_j in self.constraints[X_i]:
            # First, get a list of all possible pairs of values between variables X_i and X_j
            self.constraints[X_i][X_j] = self.get_all_possible_pairs(
                self.domains[X_i], self.domains[X_j])

        # Next, filter this list of value pairs through the function
        # 'filter_function', so that only the legal value pairs remain
        self.constraints[X_i][X_j] = list(
            filter(lambda value_pair: filter_function(*value_pair), self.constraints[X_i][X_j]))

    def add_all_different_constraint(self, variables):
        """Add an Alldiff constraint between all of the variables in the
        list 'variables'.
        """
        for (X_i, X_j) in self.get_all_possible_pairs(variables, variables):
            if X_i != X_j:
                self.add_constraint_one_way(X_i, X_j, lambda x, y: x != y)

    def backtracking_search(self):
        """This functions starts the CSP solver and returns the found
        solution.
        """
        # Make a so-called "deep copy" of the dictionary containing the
        # domains of the CSP variables. The deep copy is required to
        # ensure that any changes made to 'assignment' does not have any
        # side effects elsewhere.
        assignment = copy.deepcopy(self.domains)

        # Run AC-3 on all constraints in the CSP, to weed out all of the
        # values that are not arc-consistent to begin with
        self.inference(assignment, self.get_all_arcs())

        # Call backtrack with the partial assignment 'assignment'
        return self.backtrack(assignment)

    def backtrack(self, assignment):
        """The function 'Backtrack' from the pseudocode in the
        textbook.

        The function is called recursively, with a partial assignment of
        values 'assignment'. 'assignment' is a dictionary that contains
        a list of all legal values for the variables that have *not* yet
        been decided, and a list of only a single value for the
        variables that *have* been decided.

        When all of the variables in 'assignment' have lists of length
        one, X_i.e. when all variables have been assigned a value, the
        function should return 'assignment'. Otherwise, the search
        should continue. When the function 'inference' is called to run
        the AC-3 algorithm, the lists of legal values in 'assignment'
        should get reduced as AC-3 discovers illegal values.

        IMPORTANT: For every iteration of the for-loop in the
        pseudocode, you need to make a deep copy of 'assignment' into a
        new variable before changing it. Every iteration of the for-loop
        should have a clean slate and not see any traces of the old
        assignments and inferences that took place in previous
        iterations of the loop.
        """
        self.n_backtracks += 1

        # if assignment is complete, X_i.e. all variables have only one legal value, return assignment
        if all(len(x) == 1 for x in assignment.values()):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in assignment[var]:
            assignment_copy = copy.deepcopy(assignment)

            assignment_copy[var] = [value]  # add {var = value} to assignment

            # inferences != failures, thus inference == True
            if self.inference(assignment_copy, self.get_all_arcs()):
                # Recursively call the backtrack function
                result = self.backtrack(assignment_copy)

                # if result != failure then return result
                if result:
                    return result

        self.n_failures += 1
        return False

    def select_unassigned_variable(self, assignment):
        """The function 'Select-Unassigned-Variable' from the pseudocode
        in the textbook. Should return the name of one of the variables
        in 'assignment' that have not yet been decided, X_i.e. whose list
        of legal values has a length greater than one.
        """
        # return the first key in assignment with more than one legal action, X_i.e. return the first variable that has not yet been decided
        for key, value in assignment.items():
            if len(value) > 1:
                return key

    def inference(self, assignment, queue):
        """The function 'AC-3' from the pseudocode in the textbook.
        'assignment' is the current partial assignment, that contains
        the lists of legal values for each undecided variable. 'queue'
        is the initial queue of arcs that should be visited.
        """

        while queue:
            X_i, X_j = queue.pop(0)

            if self.revise(assignment, X_i, X_j):

                # If no more legal for variable X_i, return false. This is a failure!
                if not assignment[X_i]:
                    return False

                # for each X_k in X_i.NEIGHBORS - {Xj}
                for X_k in self.get_all_neighboring_arcs(X_i):
                    # if not the pair we already popped (X_j), add to the queue
                    if X_k != X_j:
                        queue.append(X_k)
        return True

    def revise(self, assignment, X_i, X_j):
        """The function 'Revise' from the pseudocode in the textbook.
        'assignment' is the current partial assignment, that contains
        the lists of legal values for each undecided variable. 'X_i' and
        'X_j' specifies the arc that should be visited. If a value is
        found in variable X_i's domain that doesn't satisfy the constraint
        between X_i and X_j, the value should be deleted from X_i's list of
        legal values in 'assignment'.
        """
        revised = False
        arcs = self.constraints[X_i][X_j]

        for val in assignment[X_i]:
            delete = True
            val_pairs = [(val, y) for y in assignment[X_j]]
            for pair in val_pairs:
                if pair in arcs:
                    delete = False
                    break
            if delete:
                assignment[X_i].remove(val)
                revised = True

        return revised


def create_map_coloring_csp():
    """Instantiate a CSP representing the map coloring problem from the
    textbook. This can be useful for testing your CSP solver as you
    develop your code.
    """
    csp = CSP()
    states = ['WA', 'NT', 'Q', 'NSW', 'V', 'SA', 'T']
    edges = {'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
             'NT': ['WA', 'Q'], 'NSW': ['Q', 'V']}
    colors = ['red', 'green', 'blue']
    for state in states:
        csp.add_variable(state, colors)
    for state, other_states in edges.items():
        for other_state in other_states:
            csp.add_constraint_one_way(
                state, other_state, lambda X_i, X_j: X_i != X_j)
            csp.add_constraint_one_way(
                other_state, state, lambda X_i, X_j: X_i != X_j)
    return csp


def create_sudoku_csp(filename):
    """Instantiate a CSP representing the Sudoku board found in the text
    file named 'filename' in the current directory.
    """
    csp = CSP()
    board = list(map(lambda x: x.strip(), open(filename, 'r')))

    for row in range(9):
        for col in range(9):
            if board[row][col] == '0':
                csp.add_variable('%d-%d' % (row, col),
                                 list(map(str, range(1, 10))))
            else:
                csp.add_variable('%d-%d' % (row, col), [board[row][col]])

    for row in range(9):
        csp.add_all_different_constraint(
            ['%d-%d' % (row, col) for col in range(9)])
    for col in range(9):
        csp.add_all_different_constraint(
            ['%d-%d' % (row, col) for row in range(9)])
    for box_row in range(3):
        for box_col in range(3):
            cells = []
            for row in range(box_row * 3, (box_row + 1) * 3):
                for col in range(box_col * 3, (box_col + 1) * 3):
                    cells.append('%d-%d' % (row, col))
            csp.add_all_different_constraint(cells)

    return csp


def print_sudoku_solution(solution):
    """Convert the representation of a Sudoku solution as returned from
    the method CSP.backtracking_search(), into a human readable
    representation.
    """
    for row in range(9):
        for col in range(9):
            print(solution['%d-%d' % (row, col)][0], end=" "),
            if col == 2 or col == 5:
                print('|', end=" "),
        print("")
        if row == 2 or row == 5:
            print('------+-------+------')


if __name__ == "__main__":
    sudoku_csp = create_sudoku_csp("easy.txt")
    solution = sudoku_csp.backtracking_search()
    print(f"Failures: {sudoku_csp.n_failures}")
    print(f"Backtracks: {sudoku_csp.n_backtracks}", end="\n\n")
    print_sudoku_solution(solution)
    print()
