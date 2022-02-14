import numpy as np
from utils import check_game_over_2

WON = 1
ONGOING = 0
TIE = -1

def possible_moves(board):
    return np.where(np.array(board) == 0)[0]

def get_best_move(board):
    best_score = np.NINF
    best_move = 0
    
    for move in possible_moves(board):
        print(f"Checking {move} from {possible_moves(board)}")
        # Make move
        board[move] = 1
        if (score := minimax(board, False)) > best_score:
            best_score = score
            best_move = move
        # Revert move
        board[move] = 0

    return best_move


def minimax(board, is_maximizing):
    state, winner = check_game_over_2(board)
    if state == WON:

        return winner
    elif state == TIE:
        return 0
    
    best_score = np.NINF if is_maximizing else np.Inf

    if is_maximizing:
        for move in possible_moves(board):
            # Make move
            board[move] = 1
            best_score = np.max([best_score, minimax(board, not is_maximizing)])
            
            # Revert move
            board[move] = 0

        return best_score
    else:
        for move in possible_moves(board):
            # Make move
            board[move] = -1

            best_score = np.min([best_score, minimax(board, not is_maximizing)])
            
            # Revert move
            board[move] = 0

        return best_score


if __name__ == '__main__':
    BOARD = np.array([0, 0, -1, 1, 1, 0, 1, -1, 1])
    print(get_best_move(BOARD))