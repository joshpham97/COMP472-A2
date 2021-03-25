import argparse
from PuzzleSolver import PuzzleSolver
import timeit

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('algorithm')
    parser.add_argument('board_path')
    args = parser.parse_args()

    start = timeit.default_timer()

    puzzleSolver = PuzzleSolver(args.board_path, args.algorithm)
    puzzleSolver.run()

    stop = timeit.default_timer()

    puzzleSolver.export(stop-start)


main()
