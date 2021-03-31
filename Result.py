class Result:
    algorithm = None
    puzzle = None
    solution_length = 0
    search_path_length = 0
    solution_found = False
    cost = 0
    execution_time = 0

    def __init__(self, algorithm, puzzle, solution_length, search_path_length, solution_found, cost, execution_time):
        self.algorithm = algorithm
        self.puzzle = puzzle

        if solution_found is True:
            self.solution_length = solution_length
            self.search_path_length = search_path_length
            self.solution_found = solution_found
            self.cost = cost
            self.execution_time = execution_time
