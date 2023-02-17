# get sudoku from file and solve it with backtracking

# import sys
import time
from math import floor


def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a + b for a in A for b in B]

digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
squares = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])
units = dict((s, [u for u in unitlist if s in u])
             for s in squares)
boxes = dict((s, set(sum([units[s][2]], [])) - set([s]))
             for s in squares)
peers = dict((s, set(sum(units[s], [])) - set([s]))
             for s in squares)



def is_valid(sudoku, idx, val):
    """Check if val is valid in row, col"""
    # check row
    sudokuLines = []
    for i in range(9):
        sudokuLines.append(sudoku[:9])

    row = floor(idx/9)
    col = idx % 9

    for i in range(9):
        # check row
        if sudokuLines[row][i] == val and i != idx % 9:
            return False
        # check column
        if sudokuLines[i][col] == val and i != floor(idx/9):
            return False

    # check square
    startRow = row - (row % 3)
    startCol = col - (col % 3)
    for i in range(3):
        for j in range(3):
            # if row % 3 == 0 and col % 3 == 0:
            # check if the value is in the square
            if sudokuLines[startRow + i][startCol + j] == val:
                return False

    return True

def grid_values(grid):
    #"Convert grid into a dict of {square: char} with '0' or '.' for empties."
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))


def solve(grid):
    """Solve sudoku"""
    # find empty cell
    # print_sudoku(sudoku_part)

    #condition d'arret
    if '0' not in grid and '.' not in grid:
        return grid_values(grid)
    else :

        for i in range(len(grid)):
                if grid[i] == '0' or grid[i] == '.':
                    for val in digits:
                        if is_valid(grid, i, val):
                            grid = grid[:i] + val + grid[i + 1:]
                            solution = solve(grid)
                            if solved(solution):
                                return solution
                            else :
                                grid = grid[:i] + '0' + grid[i + 1:]
        return grid_values(grid)

################ Display as 2-D grid ################

def display(values):
    "Display these values as a 2-D grid."
    width = 1+max(len(values[s]) for s in squares)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF':
            print(line)
    print()

def from_file(filename, sep='\n'):
    "Parse a file into a list of strings, separated by sep."
    return open(filename).read().strip().split(sep)


def shuffled(seq):
    "Return a randomly shuffled copy of the input sequence."
    seq = list(seq)
    random.shuffle(seq)
    return seq


################ System test ################


def solve_all(grids, name='', showif=0.0):
    """Attempt to solve a sequence of grids. Report results.
    When showif is a number of seconds, display puzzles that take longer.
    When showif is None, don't display any puzzles."""
    def time_solve(grid):
        start = time.perf_counter()
        values = solve(grid)
        t = time.perf_counter()-start
        display(values)
        # return (t, values)
        # if showif is not None and t > showif:
        #     display(grid_values(grid))
        #     if values: display(values)
        #     print('\n\n(%.2f seconds)\n' % t)
        #     display(values)
        return (t, solved(values))
    times, results = zip(*[time_solve(grid) for grid in grids])
    N = len(grids)
    if N > 1:
        print("Solved %d of %d %s puzzles (avg %.2f secs (%d Hz), max %.2f secs)." % (
            sum(results), N, name, sum(times)/N, N/sum(times), max(times)))
    # Display puzzles that took long enough
    if showif is not None:
        def gridshow(grid): display(grid_values(grid))
        for (t, (grid, values)) in enumerate(zip(grids, results)):
            if t > 9 or (showif and t > 0 and times[t] > showif):
                print("Problem %d ( %.2f secs)" % (t+1, times[t]))
                #gridshow(grid)


def solved(values):
    "A puzzle is solved if each unit is a permutation of the digits 1 to 9."

    def unitsolved(unit): return set(values[s] for s in unit) == set(digits)

    return values is not False and all(unitsolved(unit) for unit in unitlist)


def main():
    """Main function"""
    # print("ca vien ici 1\n")
    # print(sudoku)
    solve_all(from_file("100sudoku.txt"), "100sudoku")


    # start = time.perf_counter()
    # # for each part in sudoku solve it
    #for sudoku_part in sudoku:
    #    print(sudoku_part)
    #    print("\n")
        # solve(sudoku_part)
    #     print(sudoku_part)
    #     print("\n")

    # solve(sudoku)
    # end = time.perf_counter()
    # print_sudoku(sudoku)
    # print("Time: %f" % (end - start))


if __name__ == "__main__":
    main()
