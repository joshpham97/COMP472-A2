import argparse
from PuzzleSolver import PuzzleSolver
import timeit


def main():
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('algorithm')
    parser.add_argument('board_path')
    args = parser.parse_args()

    start = timeit.default_timer()
    print("algorithm" + args.algorithm)
    print("board_path" + args.board_path)
"""

    puzzleSolver = PuzzleSolver("C:\\Users\\Dan\\Documents\\comp472\\A2\\COMP472-A2\\puzzle.txt",
                                "idp")  # args.board_path, args.algorithm)
    puzzleSolver.run()


main()
