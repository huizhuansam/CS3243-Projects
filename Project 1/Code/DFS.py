import functools
import sys
from collections import deque
from typing import List, Tuple, Set, Deque, Union

constructor_dispatch = {
    'King': lambda coordinate: King(coordinate),
    'Queen': lambda coordinate: Queen(coordinate),
    'Bishop': lambda coordinate: Bishop(coordinate),
    'Knight': lambda coordinate: Knight(coordinate),
    'Rook': lambda coordinate: Rook(coordinate),
}


# Some utility functions
def parse_line(line: str) -> List[str]:
    return line.strip().split(':')


def extract_tag(line: str) -> str:
    return parse_line(line)[0]


def extract_value(line: str) -> str:
    return parse_line(line)[1]


def extract_values(line: str) -> List[str]:
    return parse_line(line)[1].split(' ')


def string_list_to_int_list(string_list: List[str]) -> List[int]:
    return list(map(lambda string: int(string), string_list))


def sum_int_list(int_list: List[int]) -> int:
    return functools.reduce(lambda a, b: a + b, int_list)


def get_num_pieces(line: str) -> int:
    return sum_int_list(string_list_to_int_list(extract_values(line)))


def character_to_index(character: str) -> int:
    return ord(character) - ord('a')


def index_to_character(index: int) -> str:
    return chr(index + ord('a'))


def coordinate_to_tuple(coordinate: str) -> Tuple[int, int]:
    col = character_to_index(coordinate[0])
    row = int(coordinate[1:])
    return col, row


def tuple_to_coordinate(coordinate_tuple: Tuple[int, int]) -> str:
    col = coordinate_tuple[0]
    row = coordinate_tuple[1]
    return index_to_character(col) + str(row)


def coordinates_to_tuples(line: str) -> List[Tuple[int, int]]:
    return list(map(lambda coordinate: coordinate_to_tuple(coordinate), extract_values(line)))


def coordinates_to_obstacles(line: str) -> List['Obstacle']:
    obstacle_coordinates = map(lambda string: coordinate_to_tuple(string), extract_values(line))
    return list(map(lambda coordinate: Obstacle(coordinate), obstacle_coordinates))


def unwrap_line_brackets(line: str) -> List[str]:
    return line.strip().strip('[]').split(',')


def get_string_piece(line: str) -> str:
    return unwrap_line_brackets(line)[0]


def get_string_coordinate(line: str) -> str:
    return unwrap_line_brackets(line)[1]


def get_piece(line: str) -> Union['King', 'Queen', 'Bishop', 'Knight', 'Rook']:
    return constructor_dispatch.get(get_string_piece(line))(coordinate_to_tuple(get_string_coordinate(line)))


class Piece:
    def __init__(self, coordinate: Tuple[int, int]):
        self.col = coordinate[0]
        self.row = coordinate[1]
        pass

    def get_symbol(self) -> str:
        pass

    def get_capture_symbol(self) -> str:
        pass

    def set_capture_cells(self, board: 'Board'):
        pass

    def occupy_board(self, board: 'Board'):
        board.set_cell(self.col, self.row, self.get_symbol())
        pass

    def expand_up(self, board: 'Board', col: int, row: int):
        for y in range(row - 1, -1, -1):
            if not board.is_valid_capture_cell(col, y):
                break
            board.set_cell(col, y, self.get_capture_symbol())
        pass

    def expand_down(self, board: 'Board', col: int, row: int):
        for y in range(row + 1, board.rows):
            if not board.is_valid_capture_cell(col, y):
                break
            board.set_cell(col, y, self.get_capture_symbol())
        pass

    def expand_left(self, board: 'Board', col: int, row: int):
        for x in range(col - 1, -1, -1):
            if not board.is_valid_capture_cell(x, row):
                break
            board.set_cell(x, row, self.get_capture_symbol())
        pass

    def expand_right(self, board: 'Board', col: int, row: int):
        for x in range(col + 1, board.cols):
            if not board.is_valid_capture_cell(x, row):
                break
            board.set_cell(x, row, self.get_capture_symbol())
        pass

    def expand_right_up(self, board: 'Board', col: int, row: int):
        x = col + 1
        y = row - 1
        while x < board.cols and y >= 0:
            if not board.is_valid_capture_cell(x, y):
                break
            board.set_cell(x, y, self.get_capture_symbol())
            x += 1
            y -= 1
        pass

    def expand_right_down(self, board: 'Board', col: int, row: int):
        x = col + 1
        y = row + 1
        while x < board.cols and y < board.rows:
            if not board.is_valid_capture_cell(x, y):
                break
            board.set_cell(x, y, self.get_capture_symbol())
            x += 1
            y += 1
        pass

    def expand_left_up(self, board: 'Board', col: int, row: int):
        x = col - 1
        y = row - 1
        while x >= 0 and y >= 0:
            if not board.is_valid_capture_cell(x, y):
                break
            board.set_cell(x, y, self.get_capture_symbol())
            x -= 1
            y -= 1
        pass

    def expand_left_down(self, board: 'Board', col: int, row: int):
        x = col - 1
        y = row + 1
        while x >= 0 and y < board.rows:
            if not board.is_valid_capture_cell(x, y):
                break
            board.set_cell(x, y, self.get_capture_symbol())
            x -= 1
            y += 1
        pass


class Obstacle(Piece):
    def get_symbol(self):
        return 'X'


class King(Piece):
    def get_symbol(self):
        return 'K'

    def get_capture_symbol(self) -> str:
        return 'k'

    def get_valid_moves(self, board: 'Board') -> List[Tuple[int, int]]:
        moves = []
        for x in range(self.col - 1, self.col + 2):
            for y in range(self.row - 1, self.row + 2):
                if (x != self.col or y != self.row) and board.is_valid_movement_cell(x, y):
                    moves.append((x, y))
        return moves

    def set_capture_cells(self, board: 'Board'):
        for x in range(self.col - 1, self.col + 2):
            for y in range(self.row - 1, self.row + 2):
                if (x != self.col or y != self.row) and board.is_valid_capture_cell(x, y):
                    board.set_cell(x, y, self.get_capture_symbol())
        pass


class Queen(Piece):
    def get_symbol(self):
        return 'Q'

    def get_capture_symbol(self) -> str:
        return 'q'

    def set_capture_cells(self, board: 'Board'):
        self.expand_up(board, self.col, self.row)
        self.expand_down(board, self.col, self.row)
        self.expand_left(board, self.col, self.row)
        self.expand_right(board, self.col, self.row)
        self.expand_left_up(board, self.col, self.row)
        self.expand_right_down(board, self.col, self.row)
        self.expand_left_down(board, self.col, self.row)
        self.expand_right_up(board, self.col, self.row)
        pass


class Bishop(Piece):
    def get_symbol(self):
        return 'B'

    def get_capture_symbol(self) -> str:
        return 'b'

    def set_capture_cells(self, board: 'Board'):
        self.expand_right_up(board, self.col, self.row)
        self.expand_right_down(board, self.col, self.row)
        self.expand_left_up(board, self.col, self.row)
        self.expand_left_down(board, self.col, self.row)
        pass


class Knight(Piece):
    def get_symbol(self):
        return 'H'

    def get_capture_symbol(self) -> str:
        return 'h'

    def is_capture_region(self, board: 'Board', col: int, row: int) -> bool:
        return board.is_valid_capture_cell(col, row) and not (
            row == self.row or col == self.col
            or (col == self.col - 1 and row == self.row - 1)
            or (col == self.col - 2 and row == self.row - 2)
            or (col == self.col - 1 and row == self.row + 1)
            or (col == self.col - 2 and row == self.row + 2)
            or (col == self.col + 1 and row == self.row + 1)
            or (col == self.col + 1 and row == self.row - 1)
            or (col == self.col + 2 and row == self.row + 2)
            or (col == self.col + 2 and row == self.row - 2))

    def set_capture_cells(self, board: 'Board'):
        for i in range(self.col - 2, self.col + 3):
            for j in range(self.row - 2, self.row + 3):
                if self.is_capture_region(board, i, j):
                    board.set_cell(i, j, self.get_capture_symbol())
        pass


class Rook(Piece):
    def get_symbol(self):
        return 'R'

    def get_capture_symbol(self) -> str:
        return 'r'

    def set_capture_cells(self, board: 'Board'):
        self.expand_left(board, self.col, self.row)
        self.expand_right(board, self.col, self.row)
        self.expand_up(board, self.col, self.row)
        self.expand_down(board, self.col, self.row)
        pass


class Board:
    free_cell = '0'
    board_pieces = frozenset({'K', 'Q', 'B', 'H', 'R', 'X'})

    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        self.matrix = [[self.free_cell for _ in range(cols)] for _ in range(rows)]
        pass

    def print_board(self):
        for row in range(self.rows):
            print(str(row) + ('  ' if row < 10 else ' ') + "[{0}]".format(', '.join(map(str, self.matrix[row]))))
        column_label = '   ' if self.rows < 10 else '    '
        for col in range(self.cols):
            column_label += index_to_character(col) + '  '
        print(column_label + '\n')
        pass

    def set_cell(self, col: int, row: int, symbol: str):
        self.matrix[row][col] = symbol
        pass

    def populate_board(self, piece_list: List[Piece]):
        for piece in piece_list:
            piece.occupy_board(self)
        pass

    def set_capture_areas(self, piece_list: List[Piece]):
        for piece in piece_list:
            piece.set_capture_cells(self)
        pass

    def is_within_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    def is_occupied(self, col: int, row: int) -> bool:
        return self.matrix[row][col] in self.board_pieces

    def is_valid_movement_cell(self, col: int, row: int) -> bool:
        return self.is_within_bounds(col, row) and self.matrix[row][col] == self.free_cell

    def is_valid_capture_cell(self, col: int, row: int) -> bool:
        return self.is_within_bounds(col, row) and not self.is_occupied(col, row)


class State:
    goal_positions: Set[Tuple[int, int]]

    def __init__(self, playing_piece: King, coordinates: Tuple[int, int], previous_state: Union['State', None]):
        self.playing_piece = playing_piece
        self.coordinates = coordinates
        self.previous_state = previous_state
        pass

    @classmethod
    def set_goal_positions(cls, goal_positions: List[Tuple[int, int]]):
        cls.goal_positions = set(goal_positions)
        pass

    @classmethod
    def get_trace(cls, current_state: 'State') -> List[List[Tuple[str, int]]]:
        if not current_state.previous_state:
            return []
        current_trace = [[current_state.previous_state.format_coordinates(), current_state.format_coordinates()]]
        return cls.get_trace(current_state.previous_state) + current_trace

    def is_at_goal_state(self):
        return self.coordinates in self.goal_positions

    def generate_next_states(self, board: Board) -> List['State']:
        return list(map(lambda move: State(King(move), move, self), self.playing_piece.get_valid_moves(board)))

    def format_coordinates(self) -> Tuple[str, int]:
        return index_to_character(self.coordinates[0]), self.coordinates[1]

    def to_string(self) -> str:
        return str(self.coordinates)


def search(chess_board: Board, initial_state: State) -> Tuple[List[List[Tuple[str, int]]], int]:
    if initial_state.is_at_goal_state():
        return [], 1
    frontier: Deque[State] = deque([initial_state])
    visited: Set[Tuple[int, int]] = {initial_state.coordinates}
    num_nodes_explored: int = 1

    while frontier:
        current_state: State = frontier.pop()
        next_states: List[State] = current_state.generate_next_states(chess_board)
        for next_state in next_states:
            if next_state.is_at_goal_state():
                num_nodes_explored += 1
                return State.get_trace(next_state), num_nodes_explored
            elif next_state.coordinates not in visited:
                visited.add(next_state.coordinates)
                frontier.append(next_state)
                num_nodes_explored += 1

    return [], num_nodes_explored


def run_DFS() -> Tuple[List[List[Tuple[str, int]]], int]:
    num_rows: int = 0
    num_cols: int = 0
    num_obstacles: int = 0
    num_enemy_pieces: int = 0
    num_own_pieces: int = 0
    obstacles: List[Piece] = []
    enemy_pieces: List[Piece] = []
    own_pieces: List[King] = []
    goal_positions: List[Tuple[int, int]] = []

    def set_num_rows(testcase_line: str):
        nonlocal num_rows
        num_rows = int(extract_value(testcase_line))
        pass

    def set_num_cols(testcase_line: str):
        nonlocal num_cols
        num_cols = int(extract_value(testcase_line))
        pass

    def set_num_obstacles(testcase_line: str):
        nonlocal num_obstacles
        num_obstacles = int(extract_value(testcase_line))
        pass

    def set_obstacles_list(testcase_line: str):
        nonlocal obstacles
        obstacles = coordinates_to_obstacles(testcase_line)
        pass

    def set_num_enemy_pieces(testcase_line: str):
        nonlocal num_enemy_pieces
        num_enemy_pieces = get_num_pieces(testcase_line)
        pass

    def set_num_own_pieces(testcase_line: str):
        nonlocal num_own_pieces
        num_own_pieces = get_num_pieces(testcase_line)
        pass

    def set_goal_positions(testcase_line: str):
        nonlocal goal_positions
        goal_positions = coordinates_to_tuples(testcase_line)
        pass

    initializer_dispatch = {
        'Rows':
            lambda testcase_line: set_num_rows(testcase_line),
        'Cols':
            lambda testcase_line: set_num_cols(testcase_line),
        'Number of Obstacles':
            lambda testcase_line: set_num_obstacles(testcase_line),
        'Position of Obstacles (space between)':
            lambda testcase_line: set_obstacles_list(testcase_line),
        'Number of Enemy King, Queen, Bishop, Rook, Knight (space between)':
            lambda testcase_line: set_num_enemy_pieces(testcase_line),
        'Number of Own King, Queen, Bishop, Rook, Knight (space between)':
            lambda testcase_line: set_num_own_pieces(testcase_line),
        'Goal Positions (space between)':
            lambda testcase_line: set_goal_positions(testcase_line),
    }

    with open(sys.argv[1], 'r') as testcase:
        line = testcase.readline().strip()
        while line:
            tag = extract_tag(line)
            if tag == 'Starting Position of Pieces [Piece, Pos]':
                for _ in range(num_own_pieces):
                    line = testcase.readline().strip()
                    own_pieces.append(get_piece(line))
            elif tag == 'Position of Enemy Pieces':
                for _ in range(num_enemy_pieces):
                    line = testcase.readline().strip()
                    enemy_pieces.append(get_piece(line))
            else:
                if initializer_dispatch.get(tag):
                    initializer_dispatch.get(tag)(line)
            line = testcase.readline().strip()

    chess_board: Board = Board(num_cols, num_rows)
    State.set_goal_positions(goal_positions)
    playing_piece: King = own_pieces[0]
    chess_board.populate_board(obstacles)
    chess_board.populate_board(enemy_pieces)
    chess_board.set_capture_areas(enemy_pieces)
    initial_state = State(playing_piece, (playing_piece.col, playing_piece.row), None)

    return search(chess_board, initial_state)


print(run_DFS())
