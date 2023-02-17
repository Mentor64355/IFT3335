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

def parse_grid(grid):
    """Convert grid to a dict of possible values, {square: digits}, or
    return False if a contradiction is detected."""
    ## To start, every square can be any digit; then assign values from the grid.
    values = dict((s, digits) for s in squares)
    for s, d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False  ## (Fail if we can't assign d to square s.)
    return dict(zip(squares, chars))


def grid_values(grid):
    #"Convert grid into a dict of {square: char} with '0' or '.' for empties."
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

    # TODO: understand the evaluation method and use it accordingly in hill_climbing
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

# prend un dictionnaire qui contient les valeurs pour chaque carre 3x3 (char: string)
# et retourne une liste de lignes
def grid3x3toLines(grid):
    lines = [None] * 9
    for s in grid.values():
        if lines[0] is None:
            lines[0] = s[0] + s[1] + s[2]
            lines[1] = s[3] + s[4] + s[5]
            lines[2] = s[6] + s[7] + s[8]
        else:
            lines[0] += s[0] + s[1] + s[2]
            lines[1] += s[3] + s[4] + s[5]
            lines[2] += s[6] + s[7] + s[8]

    lines[3] = lines[0][9:18]
    lines[4] = lines[1][9:18]
    lines[5] = lines[2][9:18]
    lines[6] = lines[0][18:]
    lines[7] = lines[1][18:]
    lines[8] = lines[2][18:]
    lines[0] = lines[0][0:9]
    lines[1] = lines[1][0:9]
    lines[2] = lines[2][0:9]

    return lines



def evaluate(lines):

    conflicts = 0
    for line in lines :
        conflicts += (len(line) - len(set(line))) # compte les conflits dans chaque ligne

    cols = zip(*lines) # switch les lignes et les colonnes
    for col in cols :
        conflicts += (len(col) - len(set(col))) # compte les conflits dans les colonnes

    return conflicts

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


######################## Hill Climbing ###############################

## http://aima.cs.berkeley.edu/python/search.html
## http://en.wikipedia.org/wiki/Hill_climbing_algorithm
## https://github.com/aimacode/aima-pseudocode/blob/master/md/Hill-Climbing.md

def solve_hill_climbing(grid):
    return hill_climbing(gridPer3x3Square(grid))


#retourne la grille avec toutes les cases remplies, sans conflit dans un meme square
def generate_rand_grid(squareValues):

    missingValues = "123456789"

    #on enleve toutes les valeurs qui sont deja dans le carre 3x3
    for c in squareValues:
        if c in missingValues:
            missingValues = missingValues.replace(c, '')

    while missingValues != "":
        for i in range(9):
            if squareValues[i] == '0' or squareValues[i] == '.': # on remplace la 1re case vide par la premiere valeur possible
                squareValues = squareValues[:i] + missingValues[0] + squareValues[i+1:]
                missingValues = missingValues[1:] # on enleve la valeur utilisee des valeurs possibles
    return squareValues


# echange les valeurs de la case1 et de la case2 dans le square s
def switch(s, case1, case2):
    temp = s[case1]
    s = s[0:case1] + s[case2] + s[case1+1:]
    s = s[0:case2] + temp + s[case2+1:]
    return s

def switchTest():
    s = "123456789"
    case1 = 2
    case2 = 4
    assert(switch(s, case1, case2) == "125436789")

def hill_climbing(grid, x = 10):
    # """From the initial node, keep choosing the neighbor with highest value,
    # stopping when no neighbor is better."""
    # a partir de la grille initiale, on genere des paires de neighbours pour chaque square (dictionnaire square:(case1,case2))
    # Pour chaque square, on essaie d'ameliorer le resultat en switchant 2 cases
    # On garde la disposition du square avec le meilleur score
    # On évalue la nouvelle grille et sil y a moins de conflits on garde la nouvelle grille
    # Continuer un nombre de fois donné x (par défaut 10) et retourner la meilleure grille
    # Si on a pas trouvé de meilleure grille on retourne la grille initiale

    initialGrid = grid.copy()
    currentGrid = grid.copy()

    for k in range(x):
        currentScore = evaluate(grid3x3toLines(grid))
        for s in initialGrid:
            #generer les neighbours
            currentGrid[s] = generate_rand_grid(grid[s])
            neighbours = getNeighbours(initialGrid.get(s))
            while len(neighbours) > 0:
                candidate = currentGrid.copy()
                candidate[s] = switch(candidate.get(s), neighbours[-1][0], neighbours[-1][1])
                thisScore = evaluate(grid3x3toLines(candidate))
                if thisScore < currentScore:
                    currentGrid = candidate.copy()
                    currentScore = thisScore
                neighbours.pop()


    return grid_values("".join(grid3x3toLines(currentGrid)))



# prend en argument les valeurs d'un square et genere une liste de paires de cases pouvant etre switched
def getNeighbours(s):
    # TODO retourner une liste de voisant en switchant les valeur de la grille
    # (Ne PAS switcher les chiffres deja fixe) du noeud
    positions0 = [] # liste de positions ou il y  a un 0
    neighbours = [] #liste de paires de positions ou il y a un 0
    for i in range(9):
        if s[i] == '0' or s[i] == '.':
            positions0.append(i)

    # creer la liste de paires a partir des positions des 0
    for i in positions0:
        for j in positions0 :
            if i != j and (tuple([j, i]) not in neighbours):
                neighbours.append(tuple([i, j]))

    return neighbours


def neighboursTest():
    s = "123400780"
    assert(getNeighbours(s) == [(4,5), (4,8), (5,8)])


# prend en argument la grille et retourne un dictionnaire de ses valeurs pour chaque carre 3x3
# char: str
def gridPer3x3Square(grid):
    gridValues = {'0': grid[0] + grid[1] + grid[2] + grid[9] + grid[10] + grid[11] + grid[18] + grid[19] + grid[20],
                  '1': grid[3] + grid[4] + grid[5] + grid[12] + grid[13] + grid[14] + grid[21] + grid[22] + grid[23],
                  '2': grid[6] + grid[7] + grid[8] + grid[15] + grid[16] + grid[17] + grid[24] + grid[25] + grid[26],
                  '3': grid[27] + grid[28] + grid[29] + grid[36] + grid[37] + grid[38] + grid[45] + grid[46] + grid[47],
                  '4': grid[30] + grid[31] + grid[32] + grid[39] + grid[40] + grid[41] + grid[48] + grid[49] + grid[50],
                  '5': grid[33] + grid[34] + grid[35] + grid[42] + grid[43] + grid[44] + grid[51] + grid[52] + grid[53],
                  '6': grid[54] + grid[55] + grid[56] + grid[63] + grid[64] + grid[65] + grid[72] + grid[73] + grid[74],
                  '7': grid[57] + grid[58] + grid[59] + grid[66] + grid[67] + grid[68] + grid[75] + grid[76] + grid[77],
                  '8': grid[60] + grid[61] + grid[62] + grid[69] + grid[70] + grid[71] + grid[78] + grid[79] + grid[80]}


    return gridValues

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
exGrid = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"

if __name__ == '__main__':
    test()
    #switchTest()
    #neighboursTest()
    # solve_all(from_file("100sudoku.txt"), "100sudoku")
    # solve_all(from_file("top95.txt"), "top95", None)
    # solve_all(from_file("easy50.txt", '========'), "easy", None)
    # solve_all(from_file("easy50.txt", '========'), "easy", None)
    # solve_all(from_file("top95.txt"), "hard", None)
    # solve_all(from_file("hardest.txt"), "hardest", None)
    solve_all([random_puzzle() for _ in range(100)], "random", 100.0)

## References used:
## http://www.scanraid.com/BasicStrategies.htm
## http://www.sudokudragon.com/sudokustrategy.htm
## http://www.krazydad.com/blog/2005/09/29/an-index-of-sudoku-strategies/
## http://www2.warwick.ac.uk/fac/sci/moac/currentstudents/peter_cock/python/sudoku/