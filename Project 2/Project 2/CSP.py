import random
from itertools import product
import sys
from collections import deque
from typing import Tuple, Dict, Set, Callable, List, Deque

attack_memo_table: Dict[str, Dict[Tuple[int, int], Set[Tuple[int, int]]]] = {
    'King': {},
    'Queen': {},
    'Bishop': {},
    'Rook': {},
    'Knight': {},
}

enums = {'King': 0, 'Queen': 1, 'Bishop': 2, 'Rook': 3, 'Knight': 4}


def extract_tag(line: str) -> str:
    return line.split(':')[0].strip('-')


def extract_value(line: str) -> str:
    return line.split(':')[1].strip('-')


def index_to_character(index: int) -> str:
    return chr(ord('a') + index)


def character_to_index(character: str) -> int:
    return ord(character) - ord('a')


def position_to_coordinate(position: str) -> Tuple[int, int]:
    return character_to_index(position[0]), int(position[1:])


class Piece:
    string_representation: str

    def __repr__(self):
        return self.string_representation

    def get_attack_coordinates(self, board: 'Board', col: int, row: int) -> Set[Tuple[int, int]]:
        pass

    def attacks_all_possible_coordinates_from(self, col: int, row: int, piece: str, board: 'Board') -> bool:
        return len(board.get_available_coordinates(piece) - self.get_attack_coordinates(board, col, row)) == 0


class King(Piece):
    string_representation = 'King'

    def get_attack_coordinates(self, board: 'Board', col: int, row: int) -> Set[Tuple[int, int]]:
        if (col, row) not in attack_memo_table[self.string_representation]:
            deltas: List[Tuple[int, int]] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            coordinates: Set[Tuple[int, int]] = set()
            for c, r in deltas:
                if board.can_be_attacked(col + c, row + r):
                    coordinates.add((col + c, row + r))
            attack_memo_table[self.string_representation][(col, row)] = coordinates
        return attack_memo_table[self.string_representation][(col, row)]


class Queen(Piece):
    string_representation = 'Queen'

    def get_attack_coordinates(self, board: 'Board', col: int, row: int) -> Set[Tuple[int, int]]:
        if (col, row) not in attack_memo_table[self.string_representation]:
            coordinates: Set[Tuple[int, int]] = set().union(*[
                board.expand_up(col, row),
                board.expand_down(col, row),
                board.expand_left(col, row),
                board.expand_right(col, row),
                board.expand_left_up(col, row),
                board.expand_right_up(col, row),
                board.expand_left_down(col, row),
                board.expand_right_down(col, row)])
            attack_memo_table[self.string_representation][(col, row)] = coordinates
        return attack_memo_table[self.string_representation][(col, row)]


class Bishop(Piece):
    string_representation = 'Bishop'

    def get_attack_coordinates(self, board: 'Board', col: int, row: int) -> Set[Tuple[int, int]]:
        if (col, row) not in attack_memo_table[self.string_representation]:
            coordinates: Set[Tuple[int, int]] = set().union(*[
                board.expand_left_up(col, row),
                board.expand_right_up(col, row),
                board.expand_left_down(col, row),
                board.expand_right_down(col, row)])
            attack_memo_table[self.string_representation][(col, row)] = coordinates
        return attack_memo_table[self.string_representation][(col, row)]


class Rook(Piece):
    string_representation = 'Rook'

    def get_attack_coordinates(self, board: 'Board', col: int, row: int) -> Set[Tuple[int, int]]:
        if (col, row) not in attack_memo_table[self.string_representation]:
            coordinates: Set[Tuple[int, int]] = set().union(*[
                board.expand_up(col, row),
                board.expand_down(col, row),
                board.expand_left(col, row),
                board.expand_right(col, row)])
            attack_memo_table[self.string_representation][(col, row)] = coordinates
        return attack_memo_table[self.string_representation][(col, row)]


class Knight(Piece):
    string_representation = 'Knight'

    def get_attack_coordinates(self, board: 'Board', col: int, row: int) -> Set[Tuple[int, int]]:
        if (col, row) not in attack_memo_table[self.string_representation]:
            deltas = [(-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2)]
            res: Set[Tuple[int, int]] = set()
            for c, r in deltas:
                if board.can_be_attacked(c + col, r + row):
                    res.add((c + col, r + row))
            attack_memo_table[self.string_representation][(col, row)] = res
        return attack_memo_table[self.string_representation][(col, row)]


piece_dispatch = {
    'King': King(),
    'Queen': Queen(),
    'Rook': Rook(),
    'Knight': Knight(),
    'Bishop': Bishop(),
}


class Board:
    def __init__(
            self,
            cols: int,
            rows: int,
            obstacles: Set[Tuple[int, int]],
            assignment: Dict[Tuple[int, int], Piece],
            unassigned_piece_quantities: Dict[str, int],
            piece_available_coordinates: Dict[str, Set[Tuple[int, int]]]):
        self.cols = cols
        self.rows = rows
        self.obstacles = obstacles
        self.assignment = assignment
        self.unassigned_piece_quantities = unassigned_piece_quantities
        self.piece_available_coordinates = piece_available_coordinates
        pass

    def can_be_attacked(self, col: int, row: int) -> bool:
        return (col, row) not in self.obstacles and self.within_bounds(col, row)

    def within_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    def to_dict(self) -> Dict[Tuple[str, int], str]:
        return {(index_to_character(c[0]), c[1]): repr(p) for c, p in self.assignment.items()}

    def is_complete_assignment(self) -> bool:
        return len(self.unassigned_piece_quantities) == 0

    def is_consistent(self) -> bool:
        for p, s in self.piece_available_coordinates.items():
            if len(s) < self.unassigned_piece_quantities[p]:
                return False
            for a, b in self.piece_available_coordinates.items():
                if p is not a:
                    if len(s.union(b)) < self.unassigned_piece_quantities[p] + self.unassigned_piece_quantities[a]:
                        return False
        return True

    def get_available_coordinates(self, piece: str) -> Set[Tuple[int, int]]:
        return self.piece_available_coordinates[piece]

    def assign(self, piece: str, col: int, row: int) -> 'Board':
        selected_piece: Piece = piece_dispatch[piece]
        piece_attack_coordinates: Set[Tuple[int, int]] = selected_piece.get_attack_coordinates(self, col, row)
        new_assignment: Dict[Tuple[int, int], Piece] = {c: p for c, p in self.assignment.items()}
        new_assignment[(col, row)] = selected_piece
        new_unassigned_piece_quantities: Dict[str, int] = {c: n for c, n in self.unassigned_piece_quantities.items()}
        new_unassigned_piece_quantities[piece] -= 1
        new_piece_available_coordinates: Dict[str, Set[Tuple[int, int]]] = {
            p: {(c, r) for c, r in s if (c, r) != (col, row) and
                (c, r) not in piece_attack_coordinates and
                (col, row) not in piece_dispatch[p].get_attack_coordinates(self, c, r)}
            for p, s in self.piece_available_coordinates.items()
        }
        if new_unassigned_piece_quantities[piece] < 1:
            new_unassigned_piece_quantities.pop(piece)
            new_piece_available_coordinates.pop(piece)
        return Board(
            self.cols,
            self.rows,
            self.obstacles,
            new_assignment,
            new_unassigned_piece_quantities,
            new_piece_available_coordinates)

    def get_arcs(self) -> Deque[Tuple[str, str]]:
        return deque(product(self.piece_available_coordinates.keys(), self.piece_available_coordinates.keys()))

    def choose_unassigned_piece(self) -> str:
        most_restrictive_set_size: int = min(map(len, self.piece_available_coordinates.values()))
        for p, v in self.piece_available_coordinates.items():
            if len(v) == most_restrictive_set_size:
                return p

    def get_unassigned_pieces(self) -> Set[str]:
        return set(self.piece_available_coordinates.keys())

    def remove_available_coordinates(self, piece: str, coordinates: Set[Tuple[int, int]]):
        self.piece_available_coordinates[piece] -= coordinates
        pass

    def expand_up(self, col: int, row: int) -> Set[Tuple[int, int]]:
        atk: Set[Tuple[int, int]] = set()
        for y in range(row - 1, -1, -1):
            if (col, y) in self.obstacles:
                break
            if self.can_be_attacked(col, y):
                atk.add((col, y))
        return atk

    def expand_down(self, col: int, row: int) -> Set[Tuple[int, int]]:
        atk: Set[Tuple[int, int]] = set()
        for y in range(row + 1, self.rows):
            if (col, y) in self.obstacles:
                break
            if self.can_be_attacked(col, y):
                atk.add((col, y))
        return atk

    def expand_left(self, col: int, row: int) -> Set[Tuple[int, int]]:
        atk: Set[Tuple[int, int]] = set()
        for x in range(col - 1, -1, -1):
            if (x, row) in self.obstacles:
                break
            if self.can_be_attacked(x, row):
                atk.add((x, row))
        return atk

    def expand_right(self, col: int, row: int) -> Set[Tuple[int, int]]:
        atk: Set[Tuple[int, int]] = set()
        for x in range(col + 1, self.cols):
            if (x, row) in self.obstacles:
                break
            if self.can_be_attacked(x, row):
                atk.add((x, row))
        return atk

    def expand_left_up(self, col: int, row: int) -> Set[Tuple[int, int]]:
        atk: Set[Tuple[int, int]] = set()
        x = col - 1
        y = row - 1
        while x >= 0 and y >= 0:
            if (x, y) in self.obstacles:
                break
            if self.can_be_attacked(x, y):
                atk.add((x, y))
            x -= 1
            y -= 1
        return atk

    def expand_right_up(self, col: int, row: int) -> Set[Tuple[int, int]]:
        atk: Set[Tuple[int, int]] = set()
        x = col + 1
        y = row - 1
        while x < self.cols and y >= 0:
            if (x, y) in self.obstacles:
                break
            if self.can_be_attacked(x, y):
                atk.add((x, y))
            x += 1
            y -= 1
        return atk

    def expand_left_down(self, col: int, row: int) -> Set[Tuple[int, int]]:
        atk: Set[Tuple[int, int]] = set()
        x = col - 1
        y = row + 1
        while x >= 0 and y < self.rows:
            if (x, y) in self.obstacles:
                break
            if self.can_be_attacked(x, y):
                atk.add((x, y))
            x -= 1
            y += 1
        return atk

    def expand_right_down(self, col: int, row: int) -> Set[Tuple[int, int]]:
        atk: Set[Tuple[int, int]] = set()
        x = col + 1
        y = row + 1
        while x < self.cols and y < self.rows:
            if (x, y) in self.obstacles:
                break
            if self.can_be_attacked(x, y):
                atk.add((x, y))
            x += 1
            y += 1
        return atk


def backtracking_search(board: Board) -> Board:
    if board.is_complete_assignment():
        return board
    unassigned_piece: str = board.choose_unassigned_piece()
    available_coordinates = list(board.get_available_coordinates(unassigned_piece))
    random.shuffle(available_coordinates)
    for col, row in available_coordinates:
        next_board: Board = board.assign(unassigned_piece, col, row)
        if next_board.is_consistent():
            result_board: Board = backtracking_search(next_board)
            if result_board is not None:
                return result_board


def run_CSP() -> Dict[Tuple[str, int], str]:
    rows: int = 0
    cols: int = 0
    num_obstacles: int = 0
    obstacle_coordinates: Set[Tuple[int, int]] = set()
    num_pieces: List[int] = []

    def set_rows(num: int):
        nonlocal rows
        rows = num
        pass

    def set_cols(num: int):
        nonlocal cols
        cols = num
        pass

    def set_num_obstacles(num: int):
        nonlocal num_obstacles
        num_obstacles = num
        pass

    def set_obstacle_coordinates(coordinates: Set[Tuple[int, int]]):
        nonlocal obstacle_coordinates
        obstacle_coordinates = coordinates
        pass

    def set_num_pieces(nums: List[int]):
        nonlocal num_pieces
        num_pieces = nums
        pass

    parser_dispatch: Dict[str, Callable] = {
        'Rows':
            lambda l: set_rows(int(l)),
        'Cols':
            lambda l: set_cols(int(l)),
        'Number of Obstacles':
            lambda l: set_num_obstacles(int(l)),
        'Position of Obstacles (space between)':
            lambda l: set_obstacle_coordinates(set(map(position_to_coordinate, l.split()))),
        'Number of King, Queen, Bishop, Rook, Knight (space between)':
            lambda l: set_num_pieces(list(map(int, l.split()))),
    }

    testfile = sys.argv[1]
    with open(testfile, 'r') as testcase:
        for line in testcase:
            line = line.strip()
            tag = extract_tag(line)
            parser_dispatch[tag](extract_value(line))

    def generate_initially_available_coordinates() -> Set[Tuple[int, int]]:
        available_coordinates: Set[Tuple[int, int]] = set()
        for x in range(cols):
            for y in range(rows):
                if (x, y) not in obstacle_coordinates:
                    available_coordinates.add((x, y))
        return available_coordinates

    def generate_piece_quantities() -> Dict[str, int]:
        return {k: num_pieces[v] for k, v in enums.items() if num_pieces[v] > 0}

    def generate_piece_available_coordinates() -> Dict[str, Set[Tuple[int, int]]]:
        c: Dict[str, Set[Tuple[int, int]]] = {}
        if num_pieces[enums['Queen']] > 0:
            c['Queen'] = generate_initially_available_coordinates()
        if num_pieces[enums['Rook']] > 0:
            c['Rook'] = generate_initially_available_coordinates()
        if num_pieces[enums['Bishop']] > 0:
            c['Bishop'] = generate_initially_available_coordinates()
        if num_pieces[enums['Knight']] > 0:
            c['Knight'] = generate_initially_available_coordinates()
        if num_pieces[enums['King']] > 0:
            c['King'] = generate_initially_available_coordinates()
        return c

    return backtracking_search(Board(
        cols,
        rows,
        obstacle_coordinates,
        {},
        generate_piece_quantities(),
        generate_piece_available_coordinates())).to_dict()
