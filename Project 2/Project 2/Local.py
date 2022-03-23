import sys
import random
from collections import deque
from typing import Dict, Tuple, Set, Callable, Deque, List


def unwrap_brackets(line: str) -> str:
    return line.strip('[]')


def extract_tag(line: str) -> str:
    return line.split(':')[0].strip('-')


def extract_value(line: str) -> str:
    return line.split(':')[1].strip('-')


def extract_piece(line: str) -> str:
    return unwrap_brackets(line).split(',')[0]


def extract_coordinate(line: str) -> Tuple[int, int]:
    return position_to_coordinate(unwrap_brackets(line).split(',')[1])


def index_to_character(index: int) -> str:
    return chr(ord('a') + index)


def character_to_index(character: str) -> int:
    return ord(character) - ord('a')


def position_to_coordinate(position: str) -> Tuple[int, int]:
    return character_to_index(position[0]), int(position[1:])


class Piece:
    string_representation: str

    def __init__(self, col: int, row: int):
        self.col = col
        self.row = row
        pass

    def __repr__(self) -> str:
        return self.string_representation

    def attack_coordinates(self, board: 'InitialBoard') -> Set[Tuple[int, int]]:
        pass

    def is_my_coordinate(self, col: int, row: int) -> bool:
        return col == self.col and row == self.row


class King(Piece):
    string_representation = 'King'

    def attack_coordinates(self, board: 'InitialBoard') -> Set[Tuple[int, int]]:
        coordinates: Set[Tuple[int, int]] = set()
        for col in range(self.col - 1, self.col + 2):
            for row in range(self.row - 1, self.row + 2):
                if not self.is_my_coordinate(col, row) and board.can_be_attacked(col, row):
                    coordinates.add((col, row))
        return coordinates


class Queen(Piece):
    string_representation = 'Queen'

    def attack_coordinates(self, board: 'InitialBoard') -> Set[Tuple[int, int]]:
        return set().union(
            *[board.expand_up(self.col, self.row),
              board.expand_down(self.col, self.row),
              board.expand_left(self.col, self.row),
              board.expand_right(self.col, self.row),
              board.expand_left_up(self.col, self.row),
              board.expand_right_up(self.col, self.row),
              board.expand_left_down(self.col, self.row),
              board.expand_right_down(self.col, self.row)])


class Bishop(Piece):
    string_representation = 'Bishop'

    def attack_coordinates(self, board: 'InitialBoard') -> Set[Tuple[int, int]]:
        return set().union(
            *[board.expand_left_up(self.col, self.row),
              board.expand_right_up(self.col, self.row),
              board.expand_left_down(self.col, self.row),
              board.expand_right_down(self.col, self.row)])


class Knight(Piece):
    string_representation = 'Knight'

    def attack_coordinates(self, chess_board: 'InitialBoard') -> Set[Tuple[int, int]]:
        deltas = [(-2, -1), (-2, +1), (+2, -1), (+2, +1), (-1, -2), (-1, +2), (+1, -2), (+1, +2)]
        res: Set[Tuple[int, int]] = set()
        for delta in deltas:
            if chess_board.can_be_attacked(delta[0] + self.col, delta[1] + self.row):
                res.add((delta[0] + self.col, delta[1] + self.row))
        return res


class Rook(Piece):
    string_representation = 'Rook'

    def attack_coordinates(self, board: 'InitialBoard') -> Set[Tuple[int, int]]:
        return set().union(
            *[board.expand_up(self.col, self.row),
              board.expand_down(self.col, self.row),
              board.expand_left(self.col, self.row),
              board.expand_right(self.col, self.row)])


class InitialBoard:
    def __init__(self, col: int, row: int, obstacles: Set[Tuple[int, int]], pieces: Dict[Tuple[int, int], Piece]):
        self.col = col
        self.row = row
        self.obstacles = obstacles
        self.pieces = pieces
        pass

    def create_adjacency_list(self) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        adjacency_list: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        for coordinate, piece in self.pieces.items():
            adjacent_coordinates: Set[Tuple[int, int]] = piece.attack_coordinates(self)
            if coordinate not in adjacency_list:
                adjacency_list[coordinate] = set()
            for adjacent_coordinate in adjacent_coordinates:
                adjacency_list[coordinate].add(adjacent_coordinate)
                if adjacent_coordinate not in adjacency_list:
                    adjacency_list[adjacent_coordinate] = set()
                adjacency_list[adjacent_coordinate].add(coordinate)
        for k, v in adjacency_list.items():
            v.discard(k)
        return adjacency_list

    def can_be_attacked(self, col: int, row: int) -> bool:
        return (col, row) in self.pieces

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
        for y in range(row + 1, self.row):
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
        for x in range(col + 1, self.col):
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
        while x < self.col and y >= 0:
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
        while x >= 0 and y < self.row:
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
        while x < self.col and y < self.row:
            if (x, y) in self.obstacles:
                break
            if self.can_be_attacked(x, y):
                atk.add((x, y))
            x += 1
            y += 1
        return atk


class PlayBoard:
    def __init__(self, pieces: Dict[Tuple[int, int], Piece], adjacency_list: Dict[Tuple[int, int], Set[Tuple[int, int]]]):
        self.pieces = pieces
        self.adjacency_list = adjacency_list
        pass

    def evaluate(self) -> int:
        res: int = 0
        for s in self.adjacency_list.values():
            res += len(s)
        return res

    def to_dict(self) -> Dict[Tuple[str, int], str]:
        return {(index_to_character(c[0]), c[1]): repr(self.pieces[c]) for c in self.adjacency_list.keys()}

    def generate_next_random_board(self) -> 'PlayBoard':
        randomly_selected_coordinate: Tuple[int, int] = random.choice(list(self.adjacency_list))
        return PlayBoard(self.pieces, {
            c: {v for v in s if v != randomly_selected_coordinate}
            for c, s in self.adjacency_list.items() if c != randomly_selected_coordinate})


class State:
    def __init__(self, board: PlayBoard, k: int, num_pieces: int):
        self.board = board
        self.k = k
        self.num_pieces = num_pieces
        self.value = board.evaluate()
        pass

    def __lt__(self, other: 'State') -> bool:
        return self.value < other.value

    def is_goal(self) -> bool:
        return self.value == 0 and not self.failed()

    def failed(self) -> bool:
        return self.num_pieces < self.k

    def get_dict(self) -> Dict[Tuple[str, int], str]:
        return self.board.to_dict()

    def generate_next_state(self) -> 'State':
        return State(self.board.generate_next_random_board(), self.k, self.num_pieces - 1)


def search(initial_state: State, beam_size: int) -> Dict[Tuple[str, int], str]:
    if initial_state.is_goal():
        return initial_state.get_dict()
    while True:
        beam: Deque[State] = deque()
        while len(beam) < beam_size:
            new_state: State = initial_state.generate_next_state()
            if new_state.is_goal():
                return new_state.get_dict()
            if new_state.failed():
                continue
            beam.append(new_state)

        while beam:
            current_state: State = beam.popleft()
            next_states: List[State] = []
            for _ in range(beam_size):
                next_state: State = current_state.generate_next_state()
                if next_state.is_goal():
                    return next_state.get_dict()
                if next_state.failed():
                    continue
                next_states.append(next_state)
            if len(next_states) > 0:
                best_state: State = min(next_states)
                if best_state.value < current_state.value:
                    beam.append(best_state)


def run_local() -> Dict[Tuple[str, int], str]:
    rows: int = 0
    cols: int = 0
    num_obstacles: int = 0
    obstacle_coordinates: Set[Tuple[int, int]] = set()
    k: int = 0
    num_pieces: int = 0
    piece_positions: Dict[Tuple[int, int], Piece] = {}

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

    def set_k(num: int):
        nonlocal k
        k = num
        pass

    def set_num_pieces(num: int):
        nonlocal num_pieces
        num_pieces = num
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
        'K (Minimum number of pieces left in goal)':
            lambda l: set_k(int(l)),
        'Number of King, Queen, Bishop, Rook, Knight (space between)':
            lambda l: set_num_pieces(sum(map(int, l.split()))),
    }

    piece_constructor_dispatch: Dict[str, Callable] = {
        'King': lambda c, r: King(c, r),
        'Queen': lambda c, r: Queen(c, r),
        'Bishop': lambda c, r: Bishop(c, r),
        'Knight': lambda c, r: Knight(c, r),
        'Rook': lambda c, r: Rook(c, r),
    }

    testfile = sys.argv[1]
    with open(testfile, 'r') as testcase:
        for line in testcase:
            line = line.strip()
            tag = extract_tag(line)
            if tag in parser_dispatch:
                parser_dispatch[tag](extract_value(line))
            else:
                for piece_line in testcase:
                    piece_line = piece_line.strip()
                    col, row = extract_coordinate(piece_line)
                    piece = extract_piece(piece_line)
                    piece_positions[(col, row)] = piece_constructor_dispatch[piece](col, row)

    initial_board: InitialBoard = InitialBoard(cols, rows, obstacle_coordinates, piece_positions)
    play_board: PlayBoard = PlayBoard(piece_positions, initial_board.create_adjacency_list())
    initial_state: State = State(play_board, k, num_pieces)
    return search(initial_state, k)
