import numpy as np

WON = 1
ONGOING = 0
TIE = -1


def check_game_over_2(board):
    """
    Similar to `check_game_over` but it dedects a win before the game ends
    WON = 1
    ONGOING = 0
    TIE = -1
    :param board: The board to evaluate
    :return Tuple(1,0,-1), winner)
    """

    WON = 1
    ONGOING = 0
    TIE = -1

    board = np.array(board).reshape((3,3))
    for player in [-1, 1]:
        # From: https://stackoverflow.com/a/46802686/
        mask = board==player
        out = mask.all(0).any() | mask.all(1).any()
        out |= np.diag(mask).all() | np.diag(mask[:,::-1]).all()
        if out == True:
            return (WON, player)

    # Check for tie
    if not np.any(board == 0):
        return (TIE, 0)

    return (ONGOING, None)



def possible_moves(board):
    return np.where(np.array(board) == 0)[0]

def get_best_move(board, player_sign=1):
    """
    Computes the best move using the minimax algorithm
    :param board is the current state of the board
    :param player_sign defines the sign of the player that gets optimized
    :return the index of the best possible move
    """
    best_score = np.NINF
    best_move = 0

    # Invert the board if the player to optimize is `-1`
    board = np.array(board) * player_sign
    
    for move in possible_moves(board):
        # print(f"Checking {move} from {possible_moves(board)}")
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