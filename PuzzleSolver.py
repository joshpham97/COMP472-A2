from State import State
from Result import Result
from ast import literal_eval as make_tuple
from time import perf_counter
import math
import timeit
import itertools
from heapq import heappush, heappop, heapify


def time_limit_exceeded(limit):
    return perf_counter() < limit


def make_tuple_semicolon(puzzle):
    return puzzle.replace(";", ",")


class PuzzleSolver:
    algorithm = ""
    initial_state = None
    goal_state = []
    nodes_expanded = 0
    max_frontier_size = 0
    goal_node = None
    max_search_depth = 0
    moves = None
    board_len = 0
    board_side = 0
    start = 0
    search_path = []
    results = []

    puzzle_list = list()

    def __init__(self, path):

        f = open(path, "r")
        inputText = f.readlines()
        f.close()

        # remove whitespace
        inputText = [x.strip() for x in inputText]

        for random_puzzle in inputText:
            puzzle = list()
            inputTuple = make_tuple(make_tuple_semicolon(random_puzzle))
            for row in inputTuple:
                for col in row:
                    puzzle.append(col)
            self.puzzle_list.append(puzzle)

    def reset_state(self):
        self.initial_state = None
        self.goal_state = []
        self.nodes_expanded = 0
        self.max_frontier_size = 0
        self.goal_node = None
        self.max_search_depth = 0
        self.moves = None
        self.board_len = 0
        self.board_side = 0
        self.start = 0
        self.search_path = []

    def set_new_state(self, puzzle):
        self.initial_state = puzzle
        self.board_len = len(self.initial_state)
        self.board_side = int(self.board_len ** 0.5)
        self.goal_state = [number for number in range(1, self.board_len + 1)]

    def run(self):
        for puzzle in self.puzzle_list:
            for i in range(5):
                self.reset_state()
                self.start = timeit.default_timer()
                self.set_new_state(puzzle)
                file_postfix = ""
                algorithm = ""
                if i == 0:
                    self.dfs()
                    file_postfix = "-dfs"
                    algorithm = "dfs"
                elif i == 1:
                    self.idp()
                    file_postfix = "-idp"
                    algorithm = "idp"
                elif i == 2:
                    self.ast("h1")
                    file_postfix = "-ast_h1(manhattan)"
                    algorithm = "h1"
                elif i == 3:
                    self.ast("h2")
                    file_postfix = "-ast_h2(euclidean_distances)"
                    algorithm = "h2"
                elif i == 4:
                    self.ast("h3")
                    file_postfix = "-ast_h3(out_row_column)"
                    algorithm = "h3"
                solution_name = self.flatten_puzzle(puzzle, file_postfix)

                self.export(solution_name, algorithm)

        self.generate_statistics();

    def run_algorithm_by_name(self, algorithm):
        for puzzle in self.puzzle_list:
            self.reset_state()
            self.start = timeit.default_timer()
            self.set_new_state(puzzle)
            file_postfix = ""
            algorithm = ""

            if algorithm == "dfs":
                self.dfs()
                file_postfix = "-dfs"
            elif algorithm == "idp":
                self.idp()
                file_postfix = "-idp"
            elif algorithm == "h1":
                self.ast("h1")
                file_postfix = "-ast_h1(manhattan)"
            elif algorithm == "h2":
                self.ast("h2")
                file_postfix = "-ast_h2(euclidean_distances)"
            elif algorithm == "h4":
                self.ast("h3")
                file_postfix = "-ast_h3(out_row_column)"
            solution_name = self.flatten_puzzle(puzzle, file_postfix)

            self.export(solution_name, algorithm)

        self.generate_statistics();

    def dfs(self, threshold=0):
        explored, stack = set(), list([State(self.initial_state, None, None, 0, 0, 0, 0)])

        limit = perf_counter() + 60
        # file = open('search_path.txt', 'a')
        while stack and time_limit_exceeded(limit):
            node = stack.pop()
            number_moved = "Initial puzzle" if node.number_moved is None else str(node.number_moved)
            move_to = "" if node.move == 0 else str(node.move)

            self.search_path.append(self.flatten_puzzle(node.state, "") + " " + number_moved + " " + move_to)
            # print(node.state)
            # file.write(''.join([str(elem) for elem in node.state]) + "\n")
            explored.add(node.map)

            # If the state is the goal state
            if node.state == self.goal_state:
                self.goal_node = node
                return stack

            # If there is no threshold, or we have not yet reach the threshold, we just continue
            if threshold <= 0 or self.max_search_depth < threshold:
                neighbors = reversed(self.expand(node))  # reverse pushed into the stack
            else:
                neighbors = list()

            # For each neighboring states, we check whether it is already explored or not
            for neighbor in neighbors:
                # If the state is not explored, or not already in the queue
                if neighbor.map not in explored:
                    stack.append(neighbor)
                    explored.add(neighbor.map)

                    # Compute the current depth
                    if neighbor.depth > self.max_search_depth:
                        self.max_search_depth += 1

            # how many nodes were developed (expended)
            if len(stack) > self.max_frontier_size:
                self.max_frontier_size = len(stack)

        # file.close()

    def idp(self):
        # Initialize the max depth to the dimension of the board
        threshold = 1
        previous_max_depth = self.max_search_depth

        limit = perf_counter() + 60

        while True and time_limit_exceeded(limit):
            # print("One iteration")
            self.dfs(threshold)
            # print("\n\n")

            # If we could not find the solution with the given threshold or we have reached the deepest level
            if self.goal_node is not None or self.max_search_depth < threshold:
                break
            threshold += 1

    def ast(self, heur):
        try:
            heuristic = getattr(self, heur)
        except AttributeError:
            raise NotImplementedError("Class `{}` does not implement `{}`".format(self.__class__.__name__, heur))
        limit = perf_counter() + 60
        explored, heap, heap_entry, counter = set(), list(), {}, itertools.count()
        # calculate the heuristic
        key = heuristic(self.initial_state)

        # create the root of our A*
        root = State(self.initial_state, None, None, None, 0, 0, key)

        # add the entry to out hashmap
        entry = (key, 0, root)
        heappush(heap, entry)

        # visited map
        heap_entry[root.map] = entry

        while heap and time_limit_exceeded(limit):

            node = heappop(heap)
            # locate the state in the node
            explored.add(node[2].map)
            number_moved = "Initial puzzle" if node[2].number_moved is None else str(node[2].number_moved)
            move_to = "" if node[2].move is None else str(node[2].move)

            self.search_path.append(self.flatten_puzzle(node[2].state, "") + " " + number_moved + " " + move_to)

            # check if we reached our goal
            if node[2].state == self.goal_state:
                self.goal_node = node[2]
                return heap

            # expends all possible move from the state
            neighbors = self.expand(node[2])

            for neighbor in neighbors:

                #check the heuristic of all children of this node
                neighbor.key = neighbor.cost + heuristic(neighbor.state)
                entry = (neighbor.key, neighbor.move, neighbor)

                # append it to our heapmap to visit if its a new entry
                if neighbor.map not in explored:

                    heappush(heap, entry)
                    explored.add(neighbor.map)
                    heap_entry[neighbor.map] = entry

                    if neighbor.depth > self.max_search_depth:
                        self.max_search_depth += 1

                #if it exist in the heap entry but has a lower cost ovewrite the old one
                elif neighbor.map in heap_entry and neighbor.key < heap_entry[neighbor.map][2].key:
                    try:
                        h_index = heap.index((heap_entry[neighbor.map][2].key,
                                              heap_entry[neighbor.map][2].move,
                                              heap_entry[neighbor.map][2]))
                        heap[int(h_index)] = entry
                    except:
                        # add to list
                        heap.append(entry)

                    heap_entry[neighbor.map] = entry

                    heapify(heap)

            if len(heap) > self.max_frontier_size:
                self.max_frontier_size = len(heap)

    def expand(self, current_state):
        self.nodes_expanded += 1

        neighbors = list()

        # Generates the next moves, however we only go right or down to prevent duplication
        for number in current_state.state:
            neighbors.append(State(self.swap(current_state.state, number, "Down"), current_state, number, "Down",
                                   current_state.depth + 1, current_state.cost + 1, 0))
            neighbors.append(State(self.swap(current_state.state, number, "Right"), current_state, number, "Right",
                                   current_state.depth + 1, current_state.cost + 1, 0))

        # Filtering out the invalid moves
        next_states = [neighbor for neighbor in neighbors if neighbor.state]

        return next_states

    def swap(self, state, number, direction):
        # Make a copy of the current state
        new_state = state[:]

        # Find the number we would like to swap
        index = state.index(number)

        # Swap the number base on the provided direction
        if direction == "Right":
            if index not in range(self.board_side - 1, self.board_len, self.board_side):
                temp = new_state[index + 1]
                new_state[index + 1] = new_state[index]
                new_state[index] = temp
                return new_state
            else:
                return None

        if direction == "Down":  # Down
            if index not in range(self.board_len - self.board_side, self.board_len):
                temp = new_state[index + self.board_side]
                new_state[index + self.board_side] = new_state[index]
                new_state[index] = temp
                return new_state
            else:
                return None

    def backtrace(self, algorithm):
        moves = list()
        current_node = self.goal_node
        cost_of_path = 0
        while self.initial_state != current_node.state:
            if algorithm == "h1":
                cost_of_path += self.h1(current_node.state)
            elif algorithm == "h2":
                cost_of_path += self.h2(current_node.state)
            elif algorithm == "h3":
                cost_of_path += self.h3(current_node.state)
            moves.insert(0, str(current_node.number_moved) + " " + current_node.move)
            current_node = current_node.parent

        return moves, cost_of_path

    def export(self, solution_file_name, algorithm):

        stop = timeit.default_timer()

        if isinstance(self.goal_node, State):
            time = stop - self.start

            # Gather statistics relating to the search
            moves, cost_of_path = self.backtrace(algorithm)
            solution_length = len(moves) + 1
            cost_of_path = cost_of_path if cost_of_path != 0 else 1 #len(moves) # For dfs, idp we use cost of 1
            search_path_length = len(self.search_path)

            # Store the results for statistic later
            self.results.append(Result(algorithm, solution_file_name, solution_length, search_path_length, True, cost_of_path, time))

            # Output solution
            file = open("./resources/" + solution_file_name + ".txt", 'w')
            file.write("path_to_goal: " + str(moves))
            file.write("\ncost_of_path: " + str(cost_of_path))
            file.write("\nnodes_expanded: " + str(self.nodes_expanded))
            file.write("\nmax_fringe_size: " + str(self.max_frontier_size))
            file.write("\nsearch_depth: " + str(self.goal_node.depth))
            file.write("\nmax_search_depth: " + str(self.max_search_depth))
            file.write("\nrunning_time: " + format(time, '.8f') + "\n\n\n")
            file.close()

            # Output search path
            with open("./resources/search_path_" + solution_file_name + ".txt", 'w') as f:
                for item in self.search_path:
                    f.write("%s\n" % item)
        else:
            search_path_length = len(self.search_path)
            self.results.append(Result(algorithm, solution_file_name, 0, search_path_length, False, 0, 60))

            file = open("./resources/" + solution_file_name + ".txt", 'w')
            file.write("puzzle: " + str(self.initial_state))
            file.write("\nNo solution")
            file.write("\nexceed_time_limit: 60sec\n\n\n")
            file.close()

            # Output search path
            file = open("./resources/search_path_" + solution_file_name + ".txt", 'w')
            file.write("\nNo solution")
            file.write("\nexceed_time_limit: 60sec\n\n\n")
            file.close()

    def generate_statistics(self):
        algorithms = ["dfs", "idp", "h1", "h2", "h3"]

        for algo in algorithms:
            total_solution_length = 0
            total_search_path_length = 0
            total_no_solution = 0
            total_with_solution = 0
            total_execution_time = 0
            total_cost = 0

            for r in self.results:
                if r.algorithm == algo:
                    if r.solution_found:
                        total_solution_length += r.solution_length
                        total_search_path_length += r.search_path_length
                        total_execution_time += r.execution_time
                        total_cost += r.cost
                        total_with_solution += 1
                    else:
                        total_no_solution += 1

            # We dont consider no solution in our average
            if total_with_solution != 0:
                average_solution_length = total_solution_length / total_with_solution
                average_search_path_length = total_search_path_length / total_with_solution
                average_execution_time = total_execution_time / total_with_solution
                average_cost = total_cost / total_with_solution
            else:
                average_solution_length = 0
                average_search_path_length = 0
                average_execution_time = 0
                average_cost = 0

            # No solution average is calculate based on the total solution
            if (total_with_solution+total_no_solution) != 0:
                average_no_solution = total_no_solution / (total_with_solution+total_no_solution)
            else:
                average_no_solution = 0

            # Export the statistics
            file = open("./resources/statistics_" + algo + ".txt", 'w')
            file.write("Total solution length: " + str(total_solution_length))
            file.write("\nAverage solution length: " + str(average_solution_length))
            file.write("\nTotal search path length: " + str(total_search_path_length))
            file.write("\nAverage search path length: " + str(average_search_path_length))
            file.write("\nTotal number of no solution " + str(total_no_solution))
            file.write("\nAverage number of no solution " + str(average_no_solution))
            file.write("\nTotal cost: " + str(total_cost))
            file.write("\nAverage cost: " + str(average_cost))
            file.write("\nTotal execution time: " + str(total_execution_time))
            file.write("\nAverage execution time: " + str(average_execution_time))
            file.close()

    def h1(self, state):
        return sum(abs(board_index % self.board_side - goal_index % self.board_side) +
                   abs(board_index // self.board_side - goal_index // self.board_side)
                   for board_index, goal_index in ((state.index(i), self.goal_state.index(i))
                                                   for i in range(1, self.board_len+1)))

    def h2(self, state):
        return sum(math.sqrt((board_index % self.board_side - goal_index % self.board_side) ** 2 +
                             (board_index / self.board_side - goal_index / self.board_side) ** 2)
                   for board_index, goal_index in ((state.index(i), self.goal_state.index(i))
                                                   for i in range(1, self.board_len+1)))

    def h3(self, state):
        return sum((board_index // self.board_side != goal_index // self.board_side) +
                   (board_index % self.board_side != goal_index % self.board_side)
                   for board_index, goal_index in ((state.index(i), self.goal_state.index(i))
                                                   for i in range(1, self.board_len+1)))



    def flatten_puzzle(self, puzzle, algo):
        flatten_str = ""
        for i, e in enumerate(puzzle):
            flatten_str += str(e)
            if (i+1) % self.board_side == 0:
                flatten_str += "_"
        return flatten_str + algo