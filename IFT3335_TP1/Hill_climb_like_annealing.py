## Solve Every Sudoku Puzzle - Question 4

## See http://norvig.com/sudoku.html

## Throughout this program we have:
##   r is a row,    e.g. 'A'
##   c is a column, e.g. '3'
##   s is a square, e.g. 'A3'
##   d is a digit,  e.g. '9'
##   u is a unit,   e.g. ['A1','B1','C1','D1','E1','F1','G1','H1','I1']
##   grid is a grid,e.g. 81 non-blank chars, e.g. starting with '.18...7...
##   values is a dict of possible values, e.g. {'A1':'12349', 'A2':'8', ...}

import random
import time
import math
from copy import deepcopy


def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a + b for a in A for b in B]


digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
squares = cross(rows, cols)
columns = [cross(rows, c) for c in cols]
lines = [cross(r, cols) for r in rows]
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])
units = dict((s, [u for u in unitlist if s in u])
             for s in squares)
boxes = dict((s, set(sum([units[s][2]], [])) - set([s]))
             for s in squares)
peers = dict((s, set(sum(units[s], [])) - set([s]))
             for s in squares)


################ Unit Tests ################

def test():
    "A set of tests that must pass."
    assert len(squares) == 81
    assert len(unitlist) == 27
    assert all(len(units[s]) == 3 for s in squares)
    assert all(len(peers[s]) == 20 for s in squares)
    assert units['C2'] == [['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2'],
                           ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
                           ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']]
    assert peers['C2'] == set(['A2', 'B2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2',
                               'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                               'A1', 'A3', 'B1', 'B3'])
    print('All tests pass.')


################ Parse a Grid ################

# def parse_grid(grid):
#     """Convert grid to a dict of possible values, {square: digits}, or
#     return False if a contradiction is detected."""
#     ## To start, every square can be any digit; then assign values from the grid.
#     values = dict((s, digits) for s in squares)
#     for s, d in grid_values(grid).items():
#         if d in digits and not assign(values, s, d):
#             return False  ## (Fail if we can't assign d to square s.)
#     return values


def parse_grid(grid):
    # To start, every square can be any digit; then assign values from the grid.
    values = dict((s, digits) for s in squares)

    # Iterate through `rows`
    for i, row in enumerate(rows):
        # Iterate through `cols`
        for j, col in enumerate(cols):
            # Get the index of the current cell
            index = i * 9 + j
            # Get the value of the current cell
            value = grid[index]

            # Format row and column into a string and save into values
            values_index = row + col
            values[values_index] = value

    return values

def grid_values(grid):
    "Convert grid into a dict of {square: char} with '0' or '.' for empties."
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))


################ Constraint Propagation ################

def assign(values, s, d):
    """Eliminate all the other values (except d) from values[s] and propagate.
    Return values, except return False if a contradiction is detected."""
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False

# def assign_random(values, s, d):
#     """Eliminate all the other values (except d) from values[s] and propagate.
#     Return values, except return False if a contradiction is detected."""
#     values[s] = d
#     return values


def eliminate(values, s, d):
    """Eliminate d from values[s]; propagate when values or places <= 2.
    Return values, except return False if a contradiction is detected."""
    if d not in values[s]:
        return values  ## Already eliminated
    values[s] = values[s].replace(d, '')
    ## (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
    if len(values[s]) == 0:
        return False  ## Contradiction: removed last value
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    ## (2) If a unit u is reduced to only one place for a value d, then put it there.
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False  ## Contradiction: no place for this value
        elif len(dplaces) == 1:
            # d can only be in one place in unit; assign it there
            if not assign(values, dplaces[0], d):
                return False
    return values

def eval_conflits(values):
    conflits = 0
    for s in squares:
        for p in peers[s]:
            if values[s] == values[p]:
                conflits += 1
    return conflits

def evaluation(values):
    """The evaluation is equal to: 0 - number of conflicts on rows and columns"""
    conflicts = 0  # Number of conflicts (-inf < conflicts =< 0)
    for line in lines:  # We run through each row
        filled_line = []
        for s in line:  # We run through the row_i
            filled_line.append(values[s])  # We append the value of box_ij
        conflicts += len(set(filled_line)) - len(rows)  # We decrement the number of unfilled box in row
    for column in columns:  # We run through each column
        filled_column = []
        for s in column:  # We run through the column_j
            filled_column.append(values[s])  # We append the value of box_ij
        conflicts += len(set(filled_column)) - len(cols)  # We decrement the number of unfilled box in column
    return conflicts


################ Display as 2-D grid ################

# def display(values):
#     "Display these values as a 2-D grid."
#     width = 1 + max(len(values[s]) for s in squares)
#     line = '+'.join(['-' * (width * 3)] * 3)
#     for r in rows:
#         print(''.join(values[r + c].center(width) + ('|' if c in '36' else ''))
#               for c in cols)
#         if r in 'CF': print(line)

def display(values):
    "Display these values as a 2-D grid."
    width = 1 + max(len(values[s]) for s in squares)
    line = '+'.join(['-' * (width * 3)] * 3)
    for r in rows:
        print(''.join(values[r + c].center(width) + ('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF':
            print(line)
    print()


######################## Hill Climbing ###############################

## http://aima.cs.berkeley.edu/python/search.html
## http://en.wikipedia.org/wiki/Hill_climbing_algorithm
## https://github.com/aimacode/aima-pseudocode/blob/master/md/Hill-Climbing.md
original_indices = None


def get_column_indices(i, type="data index"):
    """
    Get all indices for the column of ith index
    or for the ith column (depending on type)
    """
    column = i
    indices = [column + 9 * j for j in range(9)]
    return indices


def get_row_indices(self, i, type="data index"):
    """
    Get all indices for the row of ith index
    or for the ith row (depending on type)
    """
    row = i
    indices = [j + 9 * row for j in range(9)]
    return indices


def row_board_score(grid):
    score = 0
    for row in range(9):
        indices = [j + 9 * row for j in range(9)]
        row_values = [grid[i] for i in indices]
        # On convertit le tableau en set pour enlever les doublons, et le moindre de doublons qu'on a, plus le score est bas (Bas score = meilleur)
        score -= len(set(row_values))
    return score


def col_board_score(grid):
    score = 0
    for col in range(0, 9):
        indices = [col + 9 * j for j in range(9)]
        col_values = [grid[i] for i in indices]
        score -= len(set(col_values))
    return score


def board_score(grid):
    """
    Score board by viewing every row and column and giving
    -1 points for each unique entry.
    """
    score = row_board_score(grid) + col_board_score(grid)
    # Iterate from 1 to 9

    # for row in range(9):
    #     score -= len(set(self.data[self.get_row_indices(row, type="row index")]))
    # for col in range(9):
    #     score -= len(set(self.data[self.get_column_indices(col, type="column index")]))
    return score

def square_indices_with_originals(i):
    """
    Get all indices for the square of ith index
    """
    x_offset = (i // 3) * 3
    y_offset = (i % 3) * 3

    indices = [y_offset + (j % 3) + 9 * (x_offset + (j // 3)) for j in range(9)]

    return indices

def randomize_zeros(grid):
    for i in range(9):
        # Retourne tous les index du carré (y compris les indices originaux)
        sq_indices = square_indices_with_originals(i)
        # Valeurs du carré en utilisant les indices
        sq_values = [grid[ind] for ind in sq_indices]
        # On garde les indices des cases vides
        zero_indices = [ind for i, ind in enumerate(sq_indices) if sq_values[i] == 0]
        # On garde les valeurs qui ne sont pas encore dans le carré
        to_fill = [i for i in range(1, 10) if i not in sq_values]
        # On shuffle les valeurs qu'on peut utiliser
        random.shuffle(to_fill)
        # On remplace les cases vides par les valeurs qu'on a shuffle
        for ind, value in zip(zero_indices, to_fill):
            grid[ind] = value
    return grid


def generate_new_grid(grid):
    """
    Genere un nouveau tableau en changeant les valeurs de deux cases dans le meme carré
    """

    # Creer une nouvelle copie du tableau
    new_data = deepcopy(grid)

    # Choisir une carré au hasard (representé par un chiffre entre 0 et 8)
    random_square_number = random.randint(0, 8)

    # Retourne tous les index du carré qui ne sont pas dans la liste des indices originaux
    selected_square_indices = square_indices(random_square_number)
    indices_in_square = len(selected_square_indices)
    if indices_in_square < 2:
        return new_data

    # On Echange les valeurs de deux cases dans le meme carré
    random_squares = random.sample(range(indices_in_square), 2)
    first_cell_index, second_cell_index = [selected_square_indices[ind] for ind in random_squares]
    new_data[first_cell_index], new_data[second_cell_index] = new_data[second_cell_index], new_data[first_cell_index]
    return new_data

def square_indices_with_originals(i):
    """
    Get all indices for the square of ith index
    """
    x_offset = (i // 3) * 3
    y_offset = (i % 3) * 3

    indices = [y_offset + (j % 3) + 9 * (x_offset + (j // 3)) for j in range(9)]

    return indices


def square_indices(i):
    """
    Get all indices for the square of ith index
    """
    x_offset = (i // 3) * 3
    y_offset = (i % 3) * 3

    indices = [y_offset + (j % 3) + 9 * (x_offset + (j // 3)) for j in range(9)]

    # Filter out the original index
    indices = filter(lambda x: x not in original_indices, indices)
    return list(indices)

def convert_string_to_number_grid(string):
    # Convertit le string en tableau de nombres en remplaçant les points par des 0
    return [int(char) if char != '.' else 0 for char in string]


def convert_number_to_string_grid(grid):
    # Convertit le tableau de nombres en string en remplaçant les 0 par des points
    return ''.join([str(num) if num != 0 else '.' for num in grid])

def solve_hill_climbing(grid):
    grid = convert_string_to_number_grid(grid)

    # On se souvient des indices des cases qui ne sont pas vides
    global original_indices
    original_indices = [i for i in range(81) if grid[i] != 0]
    grid = randomize_zeros(grid)

    # On calcule le score de la grille: Le score correspond au doublons qu'on trouve dans les lignes et les colonnes
    # Plus le score est bas, plus la grille est bonne
    # Le sudoku est résolu quand le score est de -162) = -(9 lignes * 9 colonnes * 2)
    current_score = board_score(grid)
    best_score = current_score

    best_score = current_score
    count = 0

    while count < 10000:
        # On génère une nouvelle grille
        new_grid = generate_new_grid(grid)

        # On calcule le score de la nouvelle grille

        new_grid_score = board_score(new_grid)

        # On calcule la différence de score
        difference = current_score - new_grid_score

        # Si la difference est plus grande que 0 (donc moins de conflit), on accepte la nouvelle grille
        if difference > 0:
            grid = new_grid
            current_score = new_grid_score

        # Si le score de la grille courante (se peut que ce soit la nouvelle)
        # est meilleur que le meilleur score, on met à jour le meilleur score
        if current_score < best_score:
            best_score = current_score

        # Si le score est de -162, on a résolu le sudoku
        if new_grid_score == -162:
            print("Solved!")
            break

        count += 1

    return parse_grid(convert_number_to_string_grid(grid))




def initialize_hill_climbing(values):
    # TODO fill the 3by3 boxes with random values but without conflicts
    tmp_values = values.copy()
    l = 0
    idx = 0
    tmp = [[],[],[],[],[],[],[],[],[]]
    for s in squares:
            # print(tmp_values[i+l])
        if l == 9:
            idx += 1
            l = 0
        if len(tmp_values[s]) == 1:
            tmp[idx].append(tmp_values[s])
        l += 1
    l = 0
    idx = 0
    for s in squares:
        # fill the cell with unique random number
        if len(tmp_values[s]) > 1:

            randNum = random.choice([x for x in digits if x not in tmp[idx]])
            # while randNum in tmp[idx]:
            #     randNum = random.choice(digits)

            tmp_values[s] = randNum
            # grid[i] = randNum
            tmp[idx].append(randNum)
        l += 1
        if l == 9:
            idx += 1
            l = 0
    return tmp_values

def hill_climbing(values, initial_grid):
    """From the initial node, keep choosing the neighbor with highest value,
    stopping when no neighbor is better."""

    current_grid = values
    # while evaluation(current_grid) != 0:
    for i in range(1000):
        new_grid = swap_values(current_grid, initial_grid)
        # new_grid_val = evaluation(new_grid)
        new_grid_val = eval_conflits(new_grid)
        # curr_grid_val = evaluation(current_grid)
        curr_grid_val = eval_conflits(current_grid)

        if new_grid_val < curr_grid_val:
            current_grid = new_grid

    return current_grid


def swap_values(current_grid, initial_grid):
    """Return a list of the neighbors of the node which is obtained by exchanging the numbers of two
    squares belonging to the same box (36 possibilities per box)"""

    checked_s1_list = []
    while len(checked_s1_list) < 9:
        s1 = random.choice(squares)

        if s1 in checked_s1_list:
            continue

        checked_s1_list.append(s1)

        if len(initial_grid[s1]) == 1:
            continue

        checked_s2_list = []

        # xhose random square in same box
        s2 = random.choice(list(boxes[s1]))
        while len(initial_grid[s2]) == 1 \
                or not initial_grid[s2] in current_grid[s1] \
                or not initial_grid[s1] in current_grid[s2]:
            checked_s2_list.append(s2)

            if len(checked_s2_list) == 9:
                break

            s2 = random.choice(list(boxes[s1]))

        if len(checked_s2_list) == 9:
            continue

        # swap
        copy_current_grid = current_grid.copy()
        tmp = copy_current_grid[s1]
        copy_current_grid[s1] = copy_current_grid[s2]
        copy_current_grid[s2] = tmp

        # return copy_current_grid
        return copy_current_grid

    return current_grid




################ Utilities ################


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
        values = solve_hill_climbing(grid)
        t = time.perf_counter() - start
        ## Display puzzles that take long enough
        # if showif is not None and t > showif:
        # display(grid_values(grid))
        display(values)  # if values: display(values)
        print('(%.2f seconds)\n' % t)
        return (t, solved(values))

    times, results = zip(*[time_solve(grid) for grid in grids])
    N = len(grids)
    if N > 1:
        print("Solved %d of %d %s puzzles (avg %.2f secs (%d Hz), max %.2f secs)." % (
            sum(results), N, name, sum(times) / N, N / sum(times), max(times)))
    if showif is not None:
        def gridshow(grid):
            display(grid_values(grid))

        for (t, (grid, values)) in enumerate(zip(grids, results)):
            if t > 9 or (showif and t > 0 and times[t] > showif):
                print("Problem %d ( %.2f secs)" % (t + 1, times[t]))
                # gridshow(grid)


def solved(values):
    "A puzzle is solved if each unit is a permutation of the digits 1 to 9."

    def unitsolved(unit): return set(values[s] for s in unit) == set(digits)

    return values is not False and all(unitsolved(unit) for unit in unitlist)


def random_puzzle(N=17):
    """Make a random puzzle with N or more assignments. Restart on contradictions.
    Note the resulting puzzle is not guaranteed to be solvable, but empirically
    about 99.8% of them are solvable. Some have multiple solutions."""
    values = dict((s, digits) for s in squares)
    for s in shuffled(squares):
        if not assign(values, s, random.choice(values[s])):
            break
        ds = [values[s] for s in squares if len(values[s]) == 1]
        if len(ds) >= N and len(set(ds)) >= 8:
            return ''.join(values[s] if len(values[s]) == 1 else '.' for s in squares)
    return random_puzzle(N)  ## Give up and make a new puzzle


grid1 = '003020600900305001001806400008102900700000008006708200002609500800203009005010300'
grid2 = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'
hard1 = '.....6....59.....82....8....45........3........6..3.54...325..6..................'

if __name__ == '__main__':
    # test()
    # solve_all(from_file("100sudoku.txt"), "100sudoku", None)
    solve_all(from_file("top95.txt"), "top95", None)
    # solve_all(from_file("top95.txt"), "hard", None)
    # solve_all(from_file("hardest.txt"), "hardest", None)
    # solve_all([random_puzzle() for _ in range(99)], "random", 100.0)

## References used:
## http://www.scanraid.com/BasicStrategies.htm
## http://www.sudokudragon.com/sudokustrategy.htm
## http://www.krazydad.com/blog/2005/09/29/an-index-of-sudoku-strategies/
## http://www2.warwick.ac.uk/fac/sci/moac/currentstudents/peter_cock/python/sudoku/


