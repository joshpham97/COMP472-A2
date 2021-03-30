from State import State
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
                algorithm = ""
                if i == 0:
                    self.dfs()
                    algorithm = "dfs"
                elif i == 1:
                    self.idp()
                    algorithm = "idp"
                elif i == 2:
                    self.ast("h1")
                    algorithm = "ast_h1(manhattan)"
                elif i == 3:
                    self.ast("h2")
                    algorithm = "ast_h2(euclidean_distances)"
                elif i == 4:
                    self.ast("h3")
                    algorithm = "ast_h3(out_row_column)"
                solution_name = self.flatten_puzzle(puzzle, algorithm)
                self.export(solution_name)



    def dfs(self, threshold=0):
        explored, stack = set(), list([State(self.initial_state, None, None, 0, 0, 0, 0)])

        limit = perf_counter() + 60
        # file = open('search_path.txt', 'a')
        while stack and time_limit_exceeded(limit):
            node = stack.pop()
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

    def backtrace(self):
        moves = list()
        current_node = self.goal_node
        while self.initial_state != current_node.state:
            moves.insert(0, str(current_node.number_moved) + " " + current_node.move)
            current_node = current_node.parent

        return moves

    def export(self, solution_file_name):

        stop = timeit.default_timer()

        if isinstance(self.goal_node, State):
            time = stop - self.start
            moves = self.backtrace()

            file = open(solution_file_name, 'w')
            file.write("path_to_goal: " + str(moves))
            file.write("\ncost_of_path: " + str(len(moves)))
            file.write("\nnodes_expanded: " + str(self.nodes_expanded))
            file.write("\nmax_fringe_size: " + str(self.max_frontier_size))
            file.write("\nsearch_depth: " + str(self.goal_node.depth))
            file.write("\nmax_search_depth: " + str(self.max_search_depth))
            file.write("\nrunning_time: " + format(time, '.8f') + "\n\n\n")
            file.close()
        else:
            file = open(solution_file_name, 'w')
            file.write("puzzle: " + str(self.initial_state))
            file.write("\nexceed_time_limit: 60sec\n\n\n")
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
        return flatten_str + "-" + algo + ".txt"