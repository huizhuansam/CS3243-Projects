import functools
import math
import re
import sys
from heapq import heappop, heappush
from typing import List, Tuple, Set, Union, Dict

num_cols: int = 0
num_rows: int = 0
num_own_pieces: int = 0
num_enemy_pieces: int = 0
cost_table: List[List[int]] = [[]]
chess_grid: List[List[str]] = [[]]
goal_positions: Set[Tuple[int, int]]
obstacles: Set[str] = {'K', 'Q', 'B', 'H', 'R', 'X'}
obstacles_to_symbol: Dict[str, str] = {'King': 'K', 'Queen': 'Q', 'Bishop': 'B', 'Knight': 'H', 'Rook': 'R', 'X': 'X'}
threats: Set[str] = {'k', 'q', 'b', 'h', 'r'}
free_cell = '0'
default_cost = 1
initial_state: 'State'
own_pieces: List[Tuple[int, int]] = []
enemy_pieces: List[Tuple[str, Tuple[int, int]]] = []
obstacle_list: List[Tuple[int, int]] = []
step_costs: List[Tuple[int, Tuple[int, int]]] = []


# Parsing utilities
def parse_line(line: str) -> List[str]:
    return line.strip().split(':')


def extract_tag(line: str) -> str:
    return parse_line(line)[0]


def extract_value(line: str) -> str:
    return parse_line(line)[1]


def string_list_to_int_list(string_list: List[str]) -> List[int]:
    return list(map(lambda string: int(string), string_list))


def sum_int_list(int_list: List[int]) -> int:
    return functools.reduce(lambda a, b: a + b, int_list)


def extract_values(line: str) -> List[str]:
    return list(filter(lambda v: v != '-', parse_line(line)[1].split(' ')))


def get_num_pieces(line: str) -> int:
    return sum_int_list(string_list_to_int_list(extract_values(line)))


def index_to_character(index: int) -> str:
    return chr(index + ord('a'))


def character_to_index(character: str) -> int:
    return ord(character) - ord('a')


def unwrap_line_brackets(line: str) -> List[str]:
    return line.strip().strip('[]').split(',')


def get_string_coordinate(line: str) -> str:
    return unwrap_line_brackets(line)[1]


def coordinate_to_tuple(coordinate: str) -> Tuple[int, int]:
    col = character_to_index(coordinate[0])
    row = int(coordinate[1:])
    return col, row


def get_string_piece(line: str) -> str:
    return unwrap_line_brackets(line)[0]


def get_piece(line: str) -> Tuple[str, Tuple[int, int]]:
    return get_string_piece(line), coordinate_to_tuple(get_string_coordinate(line))


def parse_path_cost(line: str) -> Tuple[int, Tuple[int, int]]:
    return int(unwrap_line_brackets(line)[1]), coordinate_to_tuple(unwrap_line_brackets(line)[0])


def coordinates_to_tuples(line: str) -> List[Tuple[int, int]]:
    return list(map(lambda string: coordinate_to_tuple(string), extract_values(line)))


# Setters
def set_num_own_pieces(line: str):
    global num_own_pieces
    num_own_pieces = get_num_pieces(line)
    pass


def set_initial_state(coordinate: Tuple[int, int]):
    global initial_state
    col, row = coordinate
    initial_state = State(col, row, None)
    pass


def initialise_chess_grid():
    global chess_grid
    chess_grid = [[free_cell for _ in range(num_cols)] for _ in range(num_rows)]
    pass


def initialise_cost_table():
    global cost_table
    cost_table = [[default_cost for _ in range(num_cols)] for _ in range(num_rows)]
    pass


def set_cost_table():
    global cost_table
    for step_cost in step_costs:
        cost, coordinates = step_cost
        col, row = coordinates
        cost_table[row][col] = cost
    pass


def set_num_rows(line: str):
    global num_rows
    num_rows = int(extract_value(line))
    pass


def set_num_cols(line: str):
    global num_cols
    num_cols = int(extract_value(line))
    pass


def set_obstacle_list(line: str):
    global obstacle_list
    obstacle_list = coordinates_to_tuples(line)
    pass


def set_num_enemy_pieces(line: str):
    global num_enemy_pieces
    num_enemy_pieces = get_num_pieces(line)
    pass


def set_goal_positions(line: str):
    global goal_positions
    goal_positions = set(coordinates_to_tuples(line))
    pass


def set_enemies_obstacles():
    for obstacle in obstacle_list:
        col, row = obstacle
        chess_grid[row][col] = 'X'
    for piece in enemy_pieces:
        enemy, coordinate = piece
        col, row = coordinate
        chess_grid[row][col] = obstacles_to_symbol[enemy]
    pass


def expand_up(col: int, row: int, capture_symbol: str):
    for y in range(row - 1, -1, -1):
        if not is_valid_capture_cell(col, y):
            break
        chess_grid[y][col] = capture_symbol
    pass


def expand_down(col: int, row: int, capture_symbol: str):
    for y in range(row + 1, num_rows):
        if not is_valid_capture_cell(col, y):
            break
        chess_grid[y][col] = capture_symbol
    pass


def expand_left(col: int, row: int, capture_symbol: str):
    for x in range(col - 1, -1, -1):
        if not is_valid_capture_cell(x, row):
            break
        chess_grid[row][x] = capture_symbol
    pass


def expand_right(col: int, row: int, capture_symbol: str):
    for x in range(col + 1, num_cols):
        if not is_valid_capture_cell(x, row):
            break
        chess_grid[row][x] = capture_symbol
    pass


def expand_right_up(col: int, row: int, capture_symbol: str):
    x = col + 1
    y = row - 1
    while x < num_cols and y >= 0:
        if not is_valid_capture_cell(x, y):
            break
        chess_grid[y][x] = capture_symbol
        x += 1
        y -= 1
    pass


def expand_right_down(col: int, row: int, capture_symbol: str):
    x = col + 1
    y = row + 1
    while x < num_cols and y < num_rows:
        if not is_valid_capture_cell(x, y):
            break
        chess_grid[y][x] = capture_symbol
        x += 1
        y += 1
    pass


def expand_left_up(col: int, row: int, capture_symbol: str):
    x = col - 1
    y = row - 1
    while x >= 0 and y >= 0:
        if not is_valid_capture_cell(x, y):
            break
        chess_grid[y][x] = capture_symbol
        x -= 1
        y -= 1
    pass


def expand_left_down(col: int, row: int, capture_symbol: str):
    x = col - 1
    y = row + 1
    while x >= 0 and y < num_rows:
        if not is_valid_capture_cell(x, y):
            break
        chess_grid[y][x] = capture_symbol
        x -= 1
        y += 1
    pass


def set_king_threat(coordinate: Tuple[int, int]):
    col, row = coordinate
    for x in range(col - 1, col + 2):
        for y in range(row - 1, row + 2):
            if (x != col or y != row) and is_valid_capture_cell(x, y):
                chess_grid[y][x] = 'k'
    pass


def set_queen_threat(coordinate: Tuple[int, int]):
    col, row = coordinate
    expand_up(col, row, 'q')
    expand_down(col, row, 'q')
    expand_left(col, row, 'q')
    expand_right(col, row, 'q')
    expand_left_up(col, row, 'q')
    expand_right_up(col, row, 'q')
    expand_left_down(col, row, 'q')
    expand_left_up(col, row, 'q')
    pass


def set_bishop_threat(coordinate: Tuple[int, int]):
    col, row = coordinate
    expand_left_up(col, row, 'b')
    expand_right_up(col, row, 'b')
    expand_left_down(col, row, 'b')
    expand_left_up(col, row, 'b')
    pass


def set_knight_threat(coordinate: Tuple[int, int]):
    col, row = coordinate

    def is_capture_region(capture_x: int, capture_y: int) -> bool:
        return is_valid_capture_cell(capture_x, capture_y) and not (
                capture_y == row or capture_x == col
                or (capture_x == col - 1 and capture_y == row - 1)
                or (capture_x == col - 2 and capture_y == row - 2)
                or (capture_x == col - 1 and capture_y == row + 1)
                or (capture_x == col - 2 and capture_y == row + 2)
                or (capture_x == col + 1 and capture_y == row + 1)
                or (capture_x == col + 1 and capture_y == row - 1)
                or (capture_x == col + 2 and capture_y == row + 2)
                or (capture_x == col + 2 and capture_y == row - 2))

    for x in range(col - 2, col + 3):
        for y in range(row - 2, row + 3):
            if is_capture_region(x, y):
                chess_grid[y][x] = 'h'
    pass


def set_rook_threat(coordinate: Tuple[int, int]):
    col, row = coordinate
    expand_up(col, row, 'r')
    expand_down(col, row, 'r')
    expand_left(col, row, 'r')
    expand_right(col, row, 'r')
    pass


threat_setter_dispatch = {
    'King': lambda coordinate: set_king_threat(coordinate),
    'Queen': lambda coordinate: set_queen_threat(coordinate),
    'Bishop': lambda coordinate: set_bishop_threat(coordinate),
    'Knight': lambda coordinate: set_knight_threat(coordinate),
    'Rook': lambda coordinate: set_rook_threat(coordinate),
}


def set_enemies_threats():
    for piece in enemy_pieces:
        enemy, coordinate = piece
        threat_setter_dispatch.get(enemy)(coordinate)
    pass


# Board logic
def is_within_bounds(col: int, row: int) -> bool:
    return 0 <= col < num_cols and 0 <= row < num_rows


def is_occupied(col: int, row: int) -> bool:
    return chess_grid[row][col] in obstacles


def is_threatened(col: int, row: int) -> bool:
    return chess_grid[row][col] in threats


def is_valid_movement_cell(col: int, row: int) -> bool:
    return is_within_bounds(col, row) and not is_occupied(col, row) and not is_threatened(col, row)


def is_valid_capture_cell(col: int, row: int) -> bool:
    return is_within_bounds(col, row) and not is_occupied(col, row)


def get_step_cost(col: int, row: int) -> int:
    return cost_table[row][col]


def print_chess_grid():
    for row in range(num_rows):
        print(str(row) + ('  ' if row < 10 else ' ') + "[{0}]".format(', '.join(map(str, chess_grid[row]))))
    column_label = '   ' if num_rows < 10 else '    '
    for col in range(num_cols):
        column_label += index_to_character(col) + '  '
    print(column_label + '\n')
    pass


parse_utility_dispatch = {
    'Rows':
        lambda testcase_line: set_num_rows(testcase_line),
    'Cols':
        lambda testcase_line: set_num_cols(testcase_line),
    'Position of Obstacles (space between)':
        lambda testcase_line: set_obstacle_list(testcase_line),
    'Number of Enemy King, Queen, Bishop, Rook, Knight (space between)':
        lambda testcase_line: set_num_enemy_pieces(testcase_line),
    'Number of Own King, Queen, Bishop, Rook, Knight (space between)':
        lambda testcase_line: set_num_own_pieces(testcase_line),
    'Goal Positions (space between)':
        lambda testcase_line: set_goal_positions(testcase_line),
}


class State:
    def __init__(self, col: int, row: int, parent_state: Union['State', None]):
        self.col = col
        self.row = row
        self.parent_state = parent_state
        pass

    def __hash__(self) -> int:
        return hash((self.col, self.row))

    def __lt__(self, other: 'State') -> bool:
        return self.col < other.col

    def __eq__(self, other: 'State') -> bool:
        return self.col == other.col and self.row == other.row

    def at_goal_position(self) -> bool:
        return (self.col, self.row) in goal_positions

    def format_coordinate(self) -> Tuple[str, int]:
        return index_to_character(self.col), self.row

    def get_trace(self) -> List[List[Tuple[str, int]]]:
        if not self.parent_state:
            return []
        return self.parent_state.get_trace() + [[self.parent_state.format_coordinate(), self.format_coordinate()]]

    def get_cost(self):
        return get_step_cost(self.col, self.row)

    def get_next_states(self) -> List['State']:
        states: List['State'] = []
        for x in range(self.col - 1, self.col + 2):
            for y in range(self.row - 1, self.row + 2):
                if (x != self.col or y != self.row) and is_valid_movement_cell(x, y):
                    states.append(State(x, y, self))
        return states

    def get_estimate(self) -> int:
        def manhattan_dist(x: int, y: int) -> int:
            return abs(self.col - x) + abs(self.row - y)

        minimum = math.inf
        for goal_position in goal_positions:
            col, row = goal_position
            minimum = min(minimum, manhattan_dist(col, row))
        return int(minimum)


def search() -> Tuple[List[List[Tuple[str, int]]], int, int]:
    if initial_state.at_goal_position():
        return [], 1, 0

    frontier: List[Tuple[int, State]] = [(initial_state.get_estimate(), initial_state)]  # estimates
    reached: Dict[State, int] = {initial_state: 0}  # actual
    nodes_explored = 1

    while frontier:
        estimated_cost, current_state = heappop(frontier)
        actual_path_cost = reached[current_state]
        if current_state.at_goal_position():
            return current_state.get_trace(), nodes_explored, actual_path_cost
        next_states: List[State] = current_state.get_next_states()
        for next_state in next_states:
            cost_to_reach = actual_path_cost + next_state.get_cost()
            estimate = cost_to_reach + next_state.get_estimate()
            if next_state not in reached or cost_to_reach < reached[next_state]:
                nodes_explored += 1
                reached[next_state] = cost_to_reach
                heappush(frontier, (estimate, next_state))

    return [], nodes_explored, 0


def run_AStar() -> Tuple[List[List[Tuple[str, int]]], int, int]:
    with open(sys.argv[1], 'r') as testcase:
        line = testcase.readline().strip()
        while line:
            tag = extract_tag(line)
            if tag == 'Starting Position of Pieces [Piece, Pos]':
                for _ in range(num_own_pieces):
                    line = testcase.readline().strip()
                    own_pieces.append(coordinate_to_tuple(get_string_coordinate(line)))
            elif tag == 'Position of Enemy Pieces':
                for _ in range(num_enemy_pieces):
                    line = testcase.readline().strip()
                    enemy_pieces.append(get_piece(line))
            elif tag == 'Step cost to move to selected grids (Default cost is 1) [Pos, Cost]':
                line = testcase.readline().strip()
                while bool(re.search(r'\[*]', line)):
                    step_costs.append(parse_path_cost(line))
                    line = testcase.readline().strip()
                continue
            else:
                if parse_utility_dispatch.get(tag):
                    parse_utility_dispatch[tag](line)
            line = testcase.readline().strip()

    set_initial_state(own_pieces[0])
    initialise_chess_grid()
    set_enemies_obstacles()
    set_enemies_threats()
    initialise_cost_table()
    set_cost_table()

    print_chess_grid()

    return search()


print(run_AStar())
