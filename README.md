# COMP 472 Assignment 2

## Team members
Hoang Thuan Pham – 40022992 (leader)
Daniel Edery - 40062044

## Special instructions
- Our puzzles are stored in 2 files: puzzle.txt (20 3x3 puzzles), and morePuzzles.txt (puzzles used for scaling).
- Use the following code to select the puzzle file:
```
puzzleSolver = PuzzleSolver("{path to file}")  # args.board_path, args.algorithm)
```
- To run all the puzzles a puzzle file, run the following command:
```
puzzleSolver.run()
```
- To run a specific algorithm with a puzzle file, run the follwing command:
```
puzzleSolver.run_algorithm_by_name("{algorithm}")

```
where algorithm is:
```
dfs: depth-first search
idp: iterative deepening
h1: Manhattan heuristic
h2: Euclidean heuristic
h3: Number of tiles out of row + Number of tiles out of column
```
- Output will be stored in the resources folder.
- Each output file is identified by the content of the puzzle it is associated with.

## References
- Our code modifies and extends a python program that solves the 8-puzzle by Efstathios Konstantinos Peioglou.
The original code can be found at: https://github.com/speix/8-puzzle-solver.

- For our heuristics, we consulted the following lecture by Aman Dhesi.
It can be found at the following link: https://cse.iitk.ac.in/users/cs365/2009/ppt/13jan_Aman.pdf.

