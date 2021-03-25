from State import State
from ast import literal_eval as make_tuple

class PuzzleSolver:
    algorithm = ""
    initial_state = None
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    nodes_expanded = 0
    max_frontier_size = 0
    goal_node = None
    max_search_depth = 0
    moves = None
    board_len = 0
    board_side = 0

    def __init__(self, path, algorithm):
        f = open(path, "r")
        inputText = f.read()
        f.close()

        self.initial_state = list()
        inputTuple = make_tuple(inputText)
        for row in inputTuple:
            for col in row:
                self.initial_state.append(col)

        self.board_len = len(self.initial_state)
        self.board_side = int(self.board_len ** 0.5)

        self.algorithm = algorithm

    def run(self):
        if self.algorithm == "dfs":
            self.dfs()
        elif self.algorithm == "idp":
            self.idp()

    def dfs(self, threshold=0):
        explored, stack = set(), list([State(self.initial_state, None, None, 0, 0, 0, 0)])

        while stack:
            node = stack.pop()
            explored.add(node.map)

            # If the state is the goal state
            if node.state == self.goal_state:
                self.goal_node = node
                return stack

            # If there is no threshold, or we have not yet reach the threshold, we just continue
            if threshold <= 0 or self.max_search_depth < threshold:
                neighbors = reversed(self.expand(node)) # why reverse?
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

            # ???
            if len(stack) > self.max_frontier_size:
                self.max_frontier_size = len(stack)

    def idp(self):
        # Initialize the max depth to the dimension of the board
        threshold = 1
        previous_max_depth = self.max_search_depth

        while True:
            print("One iteration")
            self.dfs(threshold)

            # If we could not find the solution with the given threshold or we have reached the deepest level
            if self.goal_node is not None or self.max_search_depth < threshold:
                break

            threshold += 1

    def expand(self, current_state):
        self.nodes_expanded += 1

        neighbors = list()

        # Generates the next moves, however we only go right or down to prevent duplication
        for number in current_state.state:
            neighbors.append(State(self.swap(current_state.state, number, "Right"), current_state, number, "Right", current_state.depth + 1, current_state.cost + 1, 0))
            neighbors.append(State(self.swap(current_state.state, number, "Down"), current_state, number, "Down", current_state.depth + 1, current_state.cost + 1, 0))

        # Filtering out the invalid moves
        next_states = [neighbor for neighbor in neighbors if neighbor.state]

        return next_states

    def swap(self, state, number, direction):
        #Make a copy of the current state
        new_state = state[:]

        #Find the number we would like to swap
        index = state.index(number)

        #Swap the number base on the provided direction
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
            '''
                        if current_node.move == 1:
                            movement = 'Up'
                        elif current_node.move == 2:
                            movement = 'Down'
                        elif current_node.move == 3:
                            movement = 'Left'
                        else:
                            movement = 'Right'
            '''
            moves.insert(0, current_node.move)
            current_node = current_node.parent

        return moves

    def export(self, time):
        moves = self.backtrace()

        file = open('output.txt', 'w')
        file.write("path_to_goal: " + str(moves))
        file.write("\ncost_of_path: " + str(len(moves)))
        file.write("\nnodes_expanded: " + str(self.nodes_expanded))
        # file.write("\nfringe_size: " + str(len(frontier)))
        file.write("\nmax_fringe_size: " + str(self.max_frontier_size))
        file.write("\nsearch_depth: " + str(self.goal_node.depth))
        file.write("\nmax_search_depth: " + str(self.max_search_depth))
        file.write("\nrunning_time: " + format(time, '.8f'))
        # file.write("\nmax_ram_usage: " + format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0, '.8f'))
        file.close()
