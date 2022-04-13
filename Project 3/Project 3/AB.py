import random
from typing import Dict, Tuple, Set, Callable, List, Union
from math import inf


def studentAgent(game_board: Dict[Tuple[str, int], Tuple[str, str]]) -> Tuple[Tuple[str, int], Tuple[str, int]]:
    white_king: 'WhiteKing' = WhiteKing()  # dummy
    black_king: 'BlackKing' = BlackKing()  # dummy
    white_pieces: Dict[Piece, Tuple[int, int]] = {}
    black_pieces: Dict[Piece, Tuple[int, int]] = {}
    for position, piece_string in game_board.items():
        coordinate: Tuple[int, int] = position_to_coordinates(position)
        piece_type, color = piece_string
        piece = PIECE_CONSTRUCTOR_DISPATCH[piece_type][color]()
        if color == 'White':
            white_pieces[piece] = coordinate
            if isinstance(piece, WhiteKing):
                white_king = piece
        if color == 'Black':
            black_pieces[piece] = coordinate
            if isinstance(piece, BlackKing):
                black_king = piece
    board: 'Board' = Board(white_pieces, black_pieces)
    state: 'WhiteState' = WhiteState(board, white_king, black_king)
    return alpha_beta_pruning(state)


def alpha_beta_pruning(state: 'WhiteState') -> Tuple[Tuple[str, int], Tuple[str, int]]:
    value, move = max_value(state, -inf, inf, 3, None)
    return coordinate_to_position(move[0]), coordinate_to_position(move[1])


def max_value(state: 'WhiteState', a: float, b: float, d: int, initial_move: Union[None, Tuple[Tuple[int, int], Tuple[int, int]]]) -> Tuple[float, Tuple[Tuple[int, int], Tuple[int, int]]]:
    if d == 1:
        return state.utility_function(), initial_move
    best_value: float = -inf
    best_move: Tuple[Tuple[int, int], Tuple[int, int]] = initial_move
    for action in state.generate_all_actions():
        action_value, move = min_value(state.get_action_result(action), a, b, d - 1, initial_move if initial_move is not None else action)
        if action_value > best_value:
            best_value = action_value
            best_move = move
            a = max(a, best_value)
        if best_value >= b:
            return best_value, best_move
    return best_value, best_move


def min_value(state: 'BlackState', a: float, b: float, d: int, initial_move: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[float, Tuple[Tuple[int, int], Tuple[int, int]]]:
    if d == 0:
        return state.utility_function(), initial_move
    best_value: float = inf
    best_move: Tuple[Tuple[int, int], Tuple[int, int]] = initial_move
    for action in state.generate_all_actions():
        action_value, move = max_value(state.get_action_result(action), a, b, d - 1, initial_move)
        if action_value < best_value:
            best_value = action_value
            best_move = move
            b = min(b, best_value)
        if best_value <= a:
            return best_value, best_move
    return best_value, best_move


def index_to_character(index: int) -> str:
    return chr(ord('a') + index)


def character_to_index(character: str) -> int:
    return ord(character) - ord('a')


def coordinate_to_position(coordinate: Tuple[int, int]) -> Tuple[str, int]:
    return index_to_character(coordinate[0]), coordinate[1]


def position_to_coordinates(position: Tuple[str, int]) -> Tuple[int, int]:
    col, row = position
    return character_to_index(col), row


def is_within_bounds(col: int, row: int) -> bool:
    return 0 <= col < 5 and 0 <= row < 5


def flip(dictionary: Dict[Tuple[int, int], Set[Tuple[int, int]]]) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
    flipped: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    for move_from, destinations in dictionary.items():
        for destination in destinations:
            if destination not in flipped:
                flipped[destination] = set()
            flipped[destination].add(move_from)
    return flipped


class Piece:
    MATERIAL_VALUE: float
    THREAT_VALUE: float
    PIECE_SQUARE_TABLE: List[List[int]]
    ENDGAME_PIECE_SQUARE_TABLE: List[List[int]]

    def material_evaluation(self, board: 'Board') -> float:
        col, row = board.get_coordinate_by_piece(self)
        return self.MATERIAL_VALUE + (self.PIECE_SQUARE_TABLE[row][col] if board.count_pieces() > 8 else self.ENDGAME_PIECE_SQUARE_TABLE[row][col])

    def get_movement_coordinates(self, col: int, row: int, board: 'Board') -> Set[Tuple[int, int]]:
        pass


class King(Piece):
    MATERIAL_VALUE = 10000000
    THREAT_VALUE = 0
    DELTAS: Set[Tuple[int, int]] = {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)}

    def get_movement_coordinates(self, col: int, row: int, board: 'Board') -> Set[Tuple[int, int]]:
        movements: Set[Tuple[int, int]] = set()
        for delta_x, delta_y in self.DELTAS:
            destination_x = col + delta_x
            destination_y = row + delta_y
            if board.is_free_cell(destination_x, destination_y) or board.can_attack(destination_x, destination_y, self):
                movements.add((destination_x, destination_y))
        return movements

    def king_safety(self, board: 'Board') -> float:
        col, row = board.get_coordinate_by_piece(self)
        attacking_pieces_count: int = 0
        value_of_attacks: int = 0
        for delta_x, delta_y in self.DELTAS:
            destination_x = delta_x + col
            destination_y = delta_y + row
            if board.is_cell_threatened(destination_x, destination_y, self):
                attacking_pieces_count += 1
                num_attacked_squares = 0
                for dx, dy in self.DELTAS:
                    dst_x = dx + col
                    dst_y = dy + row
                    if board.is_cell_threatened(dst_x, dst_y, self):
                        num_attacked_squares += 1
                value_of_attacks += num_attacked_squares * max(map(lambda c: board.get_piece_by_coordinate(c).THREAT_VALUE, board.attacked_by(destination_x, destination_y, self)))

        return value_of_attacks * ATTACK_WEIGHTS[attacking_pieces_count] / 100


class WhiteKing(King):
    PIECE_SQUARE_TABLE = [[20, 30, 10, 30, 20],
                          [20, 20, 0, 20, 20],
                          [-10, -20, -20, -20, -10],
                          [-20, -30, -40, -30, -20],
                          [-30, -40, -50, -40, -30]]
    ENDGAME_PIECE_SQUARE_TABLE = [[-50, -40, -30, -40, -50],
                                  [-30, -30, 0, -30, -30],
                                  [-30, -10, 40, -10, -30],
                                  [-30, -20, 0, -20, -30],
                                  [-50, -40, -30, -40, -50]]


class BlackKing(King):
    PIECE_SQUARE_TABLE = [[-30, -40, -50, -40, -30],
                          [-20, -30, -40, -30, -20],
                          [-10, -20, -20, -20, -10],
                          [20, 20, 0, 20, 20],
                          [20, 30, 10, 30, 20]]
    ENDGAME_PIECE_SQUARE_TABLE = [[-50, -40, -30, -40, -50],
                                  [-30, -20, 0, -20, -30],
                                  [-30, -10, 40, -10, -30],
                                  [-30, -30, 0, -30, -30],
                                  [-50, -40, -30, -40, -50]]


class Queen(Piece):
    MATERIAL_VALUE = 1200
    THREAT_VALUE = 80

    def get_movement_coordinates(self, col: int, row: int, board: 'Board') -> Set[Tuple[int, int]]:
        up = board.expand_up(col, row, self)
        down = board.expand_down(col, row, self)
        left = board.expand_left(col, row, self)
        right = board.expand_right(col, row, self)
        left_up = board.expand_left_up(col, row, self)
        left_down = board.expand_left_down(col, row, self)
        right_up = board.expand_right_up(col, row, self)
        right_down = board.expand_right_down(col, row, self)
        return up.union(down).union(left).union(right).union(left_up).union(left_down).union(right_up).union(right_down)


class WhiteQueen(Queen):
    PIECE_SQUARE_TABLE = [[-20, -10, -5, -10, -20],
                          [-10, 5, 5, 5, -10],
                          [0, 5, 5, 5, -5],
                          [-10, 5, 5, 5, -10],
                          [-20, -10, -5, -10, -20]]
    ENDGAME_PIECE_SQUARE_TABLE = PIECE_SQUARE_TABLE


class BlackQueen(Queen):
    PIECE_SQUARE_TABLE = [[-20, -10, -5, -10, -20],
                          [-10, 5, 5, 5, -10],
                          [0, 5, 5, 5, -5],
                          [-10, 5, 5, 5, -10],
                          [-20, -10, -5, -10, -20]]
    ENDGAME_PIECE_SQUARE_TABLE = PIECE_SQUARE_TABLE


class Bishop(Piece):
    MATERIAL_VALUE = 400
    THREAT_VALUE = 20

    def get_movement_coordinates(self, col: int, row: int, board: 'Board') -> Set[Tuple[int, int]]:
        left_up = board.expand_left_up(col, row, self)
        left_down = board.expand_left_down(col, row, self)
        right_up = board.expand_right_up(col, row, self)
        right_down = board.expand_right_down(col, row, self)
        return left_up.union(left_down).union(right_down).union(right_up)


class WhiteBishop(Bishop):
    PIECE_SQUARE_TABLE = [[-20, -10, -10, -10, -20],
                          [-10, 5, 0, 5, -10],
                          [-10, 10, 10, 10, -10],
                          [-10, 5, 10, 5, -10],
                          [-20, -10, -10, -10, -20]]
    ENDGAME_PIECE_SQUARE_TABLE = PIECE_SQUARE_TABLE


class BlackBishop(Bishop):
    PIECE_SQUARE_TABLE = [[-20, -10, -10, -10, -20],
                          [-10, 5, 10, 5, -10],
                          [-10, 10, 10, 10, -10],
                          [-10, 5, 0, 5, -10],
                          [-20, -10, -10, -10, -20]]
    ENDGAME_PIECE_SQUARE_TABLE = PIECE_SQUARE_TABLE


class Rook(Piece):
    MATERIAL_VALUE = 600
    THREAT_VALUE = 40

    def get_movement_coordinates(self, col: int, row: int, board: 'Board') -> Set[Tuple[int, int]]:
        up = board.expand_up(col, row, self)
        down = board.expand_down(col, row, self)
        left = board.expand_left(col, row, self)
        right = board.expand_right(col, row, self)
        return up.union(down).union(left).union(right)


class WhiteRook(Rook):
    PIECE_SQUARE_TABLE = [[0, 0, 5, 0, 0],
                          [-5, 0, 0, 0, -5],
                          [-5, 0, 0, 0, -5],
                          [5, 10, 10, 10, 5],
                          [0, 0, 0, 0, 0]]
    ENDGAME_PIECE_SQUARE_TABLE = PIECE_SQUARE_TABLE


class BlackRook(Rook):
    PIECE_SQUARE_TABLE = [[0, 0, 0, 0, 0],
                          [5, 10, 10, 10, 5],
                          [-5, 0, 0, 0, -5],
                          [-5, 0, 0, 0, -5],
                          [0, 0, 5, 0, 0]]
    ENDGAME_PIECE_SQUARE_TABLE = PIECE_SQUARE_TABLE


class Knight(Piece):
    MATERIAL_VALUE = 400
    THREAT_VALUE = 20
    PIECE_SQUARE_TABLE = [[-50, -40, -30, -40, -50],
                          [-40, 10, 15, 10, -40],
                          [-30, 15, 25, 15, -30],
                          [-40, 10, 15, 10, -40],
                          [-50, -40, -30, -40, -50]]
    ENDGAME_PIECE_SQUARE_TABLE = PIECE_SQUARE_TABLE
    DELTAS: Set[Tuple[int, int]] = {(-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2)}

    def get_movement_coordinates(self, col: int, row: int, board: 'Board') -> Set[Tuple[int, int]]:
        movements: Set[Tuple[int, int]] = set()
        for delta_x, delta_y in self.DELTAS:
            destination_x = col + delta_x
            destination_y = row + delta_y
            if board.is_free_cell(destination_x, destination_y) or board.can_attack(destination_x, destination_y, self):
                movements.add((destination_x, destination_y))
        return movements


class WhiteKnight(Knight):
    pass


class BlackKnight(Knight):
    pass


class Pawn(Piece):
    MATERIAL_VALUE = 100
    THREAT_VALUE = 0
    pass


class WhitePawn(Pawn):
    PIECE_SQUARE_TABLE = [[0, 0, 0, 0, 0],
                          [5, 10, -20, 10, 5],
                          [5, 10, 25, 10, 5],
                          [50, 50, 50, 50, 50],
                          [20, 20, 20, 20, 20]]
    ENDGAME_PIECE_SQUARE_TABLE = [[0, 0, 0, 0, 0],
                                  [5, -5, -10, -5, 5],
                                  [10, 20, 30, 20, 10],
                                  [40, 40, 50, 40, 40],
                                  [60, 60, 65, 60, 60]]

    def get_movement_coordinates(self, col: int, row: int, board: 'Board') -> Set[Tuple[int, int]]:
        movements: Set[Tuple[int, int]] = set()
        capture_deltas: Set[Tuple[int, int]] = {(1, 1), (-1, 1)}
        if board.is_free_cell(col, row + 1):
            movements.add((col, row + 1))
        for delta_x, delta_y in capture_deltas:
            destination_x = delta_x + col
            destination_y = delta_y + row
            if board.can_attack(destination_x, destination_y, self):
                movements.add((destination_x, destination_y))
        return movements


class BlackPawn(Pawn):
    PIECE_SQUARE_TABLE = [[20, 20, 20, 20, 20],
                          [50, 50, 50, 50, 50],
                          [5, 10, 25, 10, 5],
                          [5, 10, -20, 10, 5],
                          [0, 0, 0, 0, 0]]
    ENDGAME_PIECE_SQUARE_TABLE = [[60, 60, 65, 60, 60],
                                  [40, 40, 50, 40, 40],
                                  [10, 20, 30, 20, 10],
                                  [5, -5, -10, -5, 5],
                                  [0, 0, 0, 0, 0]]

    def get_movement_coordinates(self, col: int, row: int, board: 'Board') -> Set[Tuple[int, int]]:
        movements: Set[Tuple[int, int]] = set()
        capture_deltas: Set[Tuple[int, int]] = {(-1, -1), (1, -1)}
        if board.is_free_cell(col, row - 1):
            movements.add((col, row - 1))
        for delta_x, delta_y in capture_deltas:
            destination_x = delta_x + col
            destination_y = delta_y + row
            if board.can_attack(destination_x, destination_y, self):
                movements.add((destination_x, destination_y))
        return movements


class Board:
    def __init__(self, white_pieces: Dict[Piece, Tuple[int, int]], black_pieces: Dict[Piece, Tuple[int, int]]):
        self.white_pieces = white_pieces
        self.black_pieces = black_pieces
        self.white_coordinates = {c: p for p, c in self.white_pieces.items()}
        self.black_coordinates = {c: p for p, c in self.black_pieces.items()}
        self.white_actions = self.get_white_actions()
        self.black_actions = self.get_black_actions()
        self.white_attack_coordinates = flip(self.white_actions)
        self.black_attack_coordinates = flip(self.black_actions)
        pass

    def attacked_by(self, col: int, row: int, piece: Piece) -> Set[Tuple[int, int]]:
        if piece in self.white_pieces:
            return self.black_attack_coordinates[(col, row)]
        return self.white_attack_coordinates[(col, row)]

    def count_pieces(self) -> int:
        return len(self.white_pieces) + len(self.black_pieces)

    def get_white_actions(self) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        actions: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        for piece, coordinate in self.white_pieces.items():
            col, row = coordinate
            actions[coordinate] = piece.get_movement_coordinates(col, row, self)
        return actions

    def get_black_actions(self) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        actions: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        for piece, coordinate in self.black_pieces.items():
            col, row = coordinate
            actions[coordinate] = piece.get_movement_coordinates(col, row, self)
        return actions

    def generate_all_white_actions(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        white_actions: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for move_from, destinations in self.white_actions.items():
            piece: Piece = self.white_coordinates[move_from]
            for destination in destinations:
                col, row = destination
                if not isinstance(piece, King) or not self.is_cell_threatened(col, row, piece):
                    white_actions.append((move_from, destination))
        random.shuffle(white_actions)
        return white_actions

    def generate_all_black_actions(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        black_actions: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for move_from, destinations in self.black_actions.items():
            piece: Piece = self.black_coordinates[move_from]
            for destination in destinations:
                col, row = destination
                if not isinstance(piece, King) or not self.is_cell_threatened(col, row, piece):
                    black_actions.append((move_from, destination))
        random.shuffle(black_actions)
        return black_actions

    def is_piece_captured(self, piece: Piece) -> bool:
        return piece not in self.white_pieces and piece not in self.black_pieces

    def material_evaluation(self) -> float:
        white_piece_values: float = sum(map(lambda x: x.material_evaluation(self), self.white_pieces.keys()))
        black_piece_values: float = sum(map(lambda x: x.material_evaluation(self), self.black_pieces.keys()))
        return white_piece_values - black_piece_values

    def is_piece_under_threat(self, piece: Piece):
        if piece in self.white_pieces:
            return self.white_pieces[piece] in self.black_attack_coordinates
        return self.black_pieces[piece] in self.white_attack_coordinates

    def is_free_cell(self, col: int, row: int) -> bool:
        return is_within_bounds(col, row) and (col, row) not in self.white_coordinates and (col, row) not in self.black_coordinates

    def can_attack(self, col: int, row: int, piece: Piece):
        if piece in self.white_pieces:
            return (col, row) in self.black_coordinates
        return (col, row) in self.white_coordinates

    def is_cell_threatened(self, col: int, row: int, piece_asking: Piece) -> bool:
        if piece_asking in self.white_pieces:
            return (col, row) in self.black_attack_coordinates
        return (col, row) in self.white_attack_coordinates

    def get_coordinate_by_piece(self, piece: Piece) -> Tuple[int, int]:
        if piece in self.white_pieces:
            return self.white_pieces[piece]
        return self.black_pieces[piece]

    def get_piece_by_coordinate(self, coordinate: Tuple[int, int]) -> Piece:
        return self.white_coordinates[coordinate] if coordinate in self.white_coordinates else self.black_coordinates[coordinate]

    def move(self, move_from: Tuple[int, int], move_to: Tuple[int, int]) -> 'Board':
        if move_from in self.white_coordinates:
            new_white_pieces = {p: (c if c != move_from else move_to) for p, c in self.white_pieces.items()}
            if move_to in self.black_coordinates:
                new_black_pieces = {p: c for p, c in self.black_pieces.items() if c != move_to}
                return Board(new_white_pieces, new_black_pieces)
            return Board(new_white_pieces, self.black_pieces)
        new_black_pieces = {p: (c if c != move_from else move_to) for p, c in self.black_pieces.items()}
        if move_to in self.white_coordinates:
            new_white_pieces = {p: c for p, c in self.white_pieces.items() if c != move_to}
            return Board(new_white_pieces, new_black_pieces)
        return Board(self.white_pieces, new_black_pieces)

    def expand_up(self, col: int, row: int, piece: Piece) -> Set[Tuple[int, int]]:
        coordinates: Set[Tuple[int, int]] = set()
        for y in range(row - 1, -1, -1):
            if self.is_free_cell(col, y):
                coordinates.add((col, y))
            elif self.can_attack(col, y, piece):
                coordinates.add((col, y))
                break
            else:
                break
        return coordinates

    def expand_down(self, col: int, row: int, piece: Piece) -> Set[Tuple[int, int]]:
        coordinates: Set[Tuple[int, int]] = set()
        for y in range(row + 1, 5):
            if self.is_free_cell(col, y):
                coordinates.add((col, y))
            elif self.can_attack(col, y, piece):
                coordinates.add((col, y))
                break
            else:
                break
        return coordinates

    def expand_left(self, col: int, row: int, piece: Piece) -> Set[Tuple[int, int]]:
        coordinates: Set[Tuple[int, int]] = set()
        for x in range(col - 1, -1, -1):
            if self.is_free_cell(x, row):
                coordinates.add((x, row))
            elif self.can_attack(x, row, piece):
                coordinates.add((x, row))
                break
            else:
                break
        return coordinates

    def expand_right(self, col: int, row: int, piece: Piece) -> Set[Tuple[int, int]]:
        coordinates: Set[Tuple[int, int]] = set()
        for x in range(col + 1, 5):
            if self.is_free_cell(x, row):
                coordinates.add((x, row))
            elif self.can_attack(x, row, piece):
                coordinates.add((x, row))
                break
            else:
                break
        return coordinates

    def expand_left_up(self, col: int, row: int, piece: Piece) -> Set[Tuple[int, int]]:
        coordinates: Set[Tuple[int, int]] = set()
        x = col - 1
        y = row - 1
        while x >= 0 and y >= 0:
            if self.is_free_cell(x, y):
                coordinates.add((x, y))
            elif self.can_attack(x, y, piece):
                coordinates.add((x, y))
                break
            else:
                break
            x = x - 1
            y = y - 1
        return coordinates

    def expand_right_up(self, col: int, row: int, piece: Piece) -> Set[Tuple[int, int]]:
        coordinates: Set[Tuple[int, int]] = set()
        x = col + 1
        y = row - 1
        while x < 5 and y >= 0:
            if self.is_free_cell(x, y):
                coordinates.add((x, y))
            elif self.can_attack(x, y, piece):
                coordinates.add((x, y))
                break
            else:
                break
            x = x + 1
            y = y - 1
        return coordinates

    def expand_left_down(self, col: int, row: int, piece: Piece) -> Set[Tuple[int, int]]:
        coordinates: Set[Tuple[int, int]] = set()
        x = col - 1
        y = row + 1
        while x >= 0 and y < 5:
            if self.is_free_cell(x, y):
                coordinates.add((x, y))
            elif self.can_attack(x, y, piece):
                coordinates.add((x, y))
                break
            else:
                break
            x = x - 1
            y = y + 1
        return coordinates

    def expand_right_down(self, col: int, row: int, piece: Piece) -> Set[Tuple[int, int]]:
        coordinates: Set[Tuple[int, int]] = set()
        x = col + 1
        y = row + 1
        while x < 5 and y < 5:
            if self.is_free_cell(x, y):
                coordinates.add((x, y))
            elif self.can_attack(x, y, piece):
                coordinates.add((x, y))
                break
            else:
                break
            x = x + 1
            y = y + 1
        return coordinates


class State:
    def __init__(self, board: Board, white_king: WhiteKing, black_king: BlackKing):
        self.board = board
        self.white_king = white_king
        self.black_king = black_king
        pass

    def utility_function(self) -> float:
        pass

    def generate_all_actions(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        pass

    def get_action_result(self, action: Tuple[Tuple[int, int], Tuple[int, int]]) -> 'State':
        pass

    def king_safety_evaluation(self) -> float:
        pass


class WhiteState(State):
    def utility_function(self) -> float:
        return self.board.material_evaluation() + self.king_safety_evaluation()

    def generate_all_actions(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        return self.board.generate_all_white_actions()

    def get_action_result(self, action: Tuple[Tuple[int, int], Tuple[int, int]]) -> 'BlackState':
        move_from, move_to = action
        return BlackState(self.board.move(move_from, move_to), self.white_king, self.black_king)

    def king_safety_evaluation(self) -> float:
        if self.board.is_piece_captured(self.black_king):
            return 10000000000
        return self.black_king.king_safety(self.board)


class BlackState(State):
    def utility_function(self) -> float:
        return -self.board.material_evaluation() - self.king_safety_evaluation()

    def generate_all_actions(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        return self.board.generate_all_black_actions()

    def get_action_result(self, action: Tuple[Tuple[int, int], Tuple[int, int]]) -> WhiteState:
        move_from, move_to = action
        return WhiteState(self.board.move(move_from, move_to), self.white_king, self.black_king)

    def king_safety_evaluation(self) -> float:
        if self.board.is_piece_captured(self.white_king):
            return -10000000000
        return self.white_king.king_safety(self.board)


INITIAL_CHESS_BOARD: Dict[Tuple[str, int], Tuple[str, str]] = {
    ('a', 1): ('Pawn', 'White'),
    ('a', 3): ('Pawn', 'Black'),
    ('b', 1): ('Pawn', 'White'),
    ('b', 3): ('Pawn', 'Black'),
    ('c', 1): ('Pawn', 'White'),
    ('c', 3): ('Pawn', 'Black'),
    ('d', 1): ('Pawn', 'White'),
    ('d', 3): ('Pawn', 'Black'),
    ('e', 1): ('Pawn', 'White'),
    ('e', 3): ('Pawn', 'Black'),
    ('a', 0): ('Rook', 'White'),
    ('a', 4): ('Rook', 'Black'),
    ('b', 0): ('Knight', 'White'),
    ('b', 4): ('Knight', 'Black'),
    ('c', 0): ('Bishop', 'White'),
    ('c', 4): ('Bishop', 'Black'),
    ('d', 0): ('Queen', 'White'),
    ('d', 4): ('Queen', 'Black'),
    ('e', 0): ('King', 'White'),
    ('e', 4): ('King', 'Black')
}
PIECE_CONSTRUCTOR_DISPATCH: Dict[str, Dict[str, Callable[[], Piece]]] = {
    'King': {
        'White': lambda: WhiteKing(),
        'Black': lambda: BlackKing()
    },
    'Queen': {
        'White': lambda: WhiteQueen(),
        'Black': lambda: BlackQueen()
    },
    'Bishop': {
        'White': lambda: WhiteBishop(),
        'Black': lambda: BlackBishop()
    },
    'Rook': {
        'White': lambda: WhiteRook(),
        'Black': lambda: BlackRook()
    },
    'Knight': {
        'White': lambda: WhiteKnight(),
        'Black': lambda: BlackKnight()
    },
    'Pawn': {
        'White': lambda: WhitePawn(),
        'Black': lambda: BlackPawn()
    },
}
ATTACK_WEIGHTS: Dict[int, int] = {
    0: 0,
    1: 0,
    2: 50,
    3: 75,
    4: 88,
    5: 94,
    6: 97,
    7: 99,
    8: 110,
    9: 200,
}
