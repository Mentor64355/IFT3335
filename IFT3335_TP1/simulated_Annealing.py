## Solve Every Sudoku Puzzle

## See http://norvig.com/sudoku.html

## Throughout this program we have:
##   r is a row,    e.g. 'A'
##   c is a column, e.g. '3'
##   s is a square, e.g. 'A3'
##   d is a digit,  e.g. '9'
##   u is a unit,   e.g. ['A1','B1','C1','D1','E1','F1','G1','H1','I1']
##   grid is a grid,e.g. 81 non-blank chars, e.g. starting with '.18...7...
##   values is a dict of possible values, e.g. {'A1':'12349', 'A2':'8', ...}

import math
from copy import deepcopy


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

peers = dict((s, set(sum(units[s], [])) - set([s]))
             for s in squares)


################ Unit Tests ################

def test():
    "A set of unit tests."
    assert len(squares) == 81
    assert len(unitlist) == 27
    assert all(len(units[s]) == 3 for s in squares)
    assert all(len(peers[s]) == 20 for s in squares)
    # assert units['C2'] == [cross('ABC', '2'), cross(
    #     'CDE', '2'), cross('CFI', '123456789')]
    # assert peers['C2'] == set('ABDEFGHI12')
    assert units['C2'] == [['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2'],
                           ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
                           ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']]
    assert peers['C2'] == set(['A2', 'B2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2',
                               'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                               'A1', 'A3', 'B1', 'B3'])
    print('All tests pass.')


################ Parse a Grid ################


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


def eliminate(values, s, d):
    """Eliminate d from values[s]; propagate when values or places <= 2.
        Return values, except return False if a contradiction is detected."""
    if d not in values[s]:
        return values  # Already eliminated
    values[s] = values[s].replace(d, '')
    # (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
    if len(values[s]) == 0:
        return False  # Contradiction: removed last value
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    # (2) If a unit u is reduced to only one place for a value d, then put it there.
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False  # Contradiction: no place for this value
        elif len(dplaces) == 1:
            # d can only be in one place in unit; assign it there
            if not assign(values, dplaces[0], d):
                return False
    return values


################ Display as 2-D grid ################

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


################ Simulated Annealing ################

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

def convert_string_to_number_grid(string):
    # Convertit le string en tableau de nombres en remplaçant les points par des 0
    return [int(char) if char != '.' else 0 for char in string]


def convert_number_to_string_grid(grid):
    # Convertit le tableau de nombres en string en remplaçant les 0 par des points
    return ''.join([str(num) if num != 0 else '.' for num in grid])


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


def solve_simulated_annealing(str_grid):
    # On convertit la grille en liste de nombres: Les points deviennent des 0
    grid = convert_string_to_number_grid(str_grid)

    # On se souvient des indices des cases qui ne sont pas vides
    global original_indices
    original_indices = [i for i in range(81) if grid[i] != 0]

    # On remplit les cases vides avec des nombres aléatoires
    grid = randomize_zeros(grid)

    # On calcule le score de la grille: Le score correspond au doublons qu'on trouve dans les lignes et les colonnes
    # Plus le score est bas, plus la grille est bonne
    # Le sudoku est résolu quand le score est de -162) = -(9 lignes * 9 colonnes * 2)
    current_score = board_score(grid)
    best_score = current_score
    temperature = 0.6
    count = 0


    while count < 10000:
        # On génère une nouvelle grille
        new_grid = generate_new_grid(grid)

        # On calcule le score de la nouvelle grille

        new_grid_score = board_score(new_grid)

        # On calcule la différence de score
        difference = current_score - new_grid_score

        # Si la difference divisée par la température est plus grande qu'un nombre aléatoire entre 0 et 1, on accepte la nouvelle grille
        if math.exp(difference / temperature) - random.random() > 0:
            grid = new_grid
            current_score = new_grid_score

        # Si le score de la grille courante (se peut que ce soit la nouvelle) est meilleur que le meilleur score, on met à jour le meilleur score
        if current_score < best_score:
            best_score = current_score

        # Si le score est de -162, on a résolu le sudoku
        if new_grid_score == -162:
            print("Solved!")
            break

        # On diminue la température
        temperature *= 0.9999
        count += 1

    return parse_grid(convert_number_to_string_grid(grid))


################ Utilities ################

def some(seq):
    "Return some element of seq that is true."
    for e in seq:
        if e: return e
    return False


def from_file(filename, sep='\n'):
    "Parse a file into a list of strings, separated by sep."
    return open(filename).read().strip().split(sep)


def shuffled(seq):
    "Return a randomly shuffled copy of the input sequence."
    seq = list(seq)
    random.shuffle(seq)
    return seq


################ System test ################

import time, random


def solve_all(grids, name='', showif=0.0):
    """Attempt to solve a sequence of grids. Report results.
    When showif is a number of seconds, display puzzles that take longer.
    When showif is None, don't display any puzzles."""

    def time_solve(grid):
        start = time.perf_counter()
        values = solve_simulated_annealing(grid)
        t = time.perf_counter() - start
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
            sum(results), N, name, sum(times) / N, N / sum(times), max(times)))
    # Display puzzles that took long enough
    if showif is not None:
        def gridshow(grid):
            display(grid_values(grid))

        for (t, (grid, values)) in enumerate(zip(grids, results)):
            if t > 9 or (showif and t > 0 and times[t] > showif):
                print("Problem %d ( %.2f secs)" % (t + 1, times[t]))
                gridshow(grid)


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
    return random_puzzle(N)  # Give up and make a new puzzle


grid1 = '003020600900305001001806400008102900700000008006708200002609500800203009005010300'
grid2 = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'
hard1 = '.....6....59.....82....8....45........3........6..3.54...325..6..................'

if __name__ == '__main__':
    test()
    # solve_all(from_file("top95.txt"), "95sudoku", None)
    # solve_all(from_file("100sudoku.txt"), "95sudoku", None)
    # solve_all(from_file("1000sudoku.txt"), "95sudoku", None)
    # solve_all(from_file("easy50.txt", '========'), "easy", None)
    # solve_all(from_file("easy50.txt", '========'), "easy", None)
    # solve_all(from_file("top95.txt"), "hard", None)
    # solve_all(from_file("hardest.txt"), "hardest", None)
    solve_all([random_puzzle() for _ in range(99)], "random", 100.0)

# inspired by
#  https://github.com/erichowens/SudokuSolver/blob/master/sudoku.py
## References used:
## https://en.wikipedia.org/wiki/Simulated_annealing
## http://www.scanraid.com/BasicStrategies.htm
## http://www.sudokudragon.com/sudokustrategy.htm
## http://www.krazydad.com/blog/2005/09/29/an-index-of-sudoku-strategies/
## http://www2.warwick.ac.uk/fac/sci/moac/currentstudents/peter_cock/python/sudoku/