import streamlit as st
import numpy as np
from utils import check_game_over_2
from minimax import get_best_move

WON = 1
ONGOING = 0
TIE = -1

board = [0] * 9
move_counter = 1
game_over = False


def show():
    st.write(
        """
        # Machine Learning - Project 3 
        ## Tic Tac Toe
        ⁣⭕❕⭕❕❌  
        ➖➕➖➕➖  
        ⭕❕⁣❌❕⭕  
        ➖➕➖➕➖  
        ❌❕❌❕⭕  
        ### Interactively play against our minimax algorithm
        """
    )
    st.write("")

    # Initialize state.
    if "board" not in st.session_state:
        st.session_state.board = [0] * 9
        st.session_state.next_player = "X"
        st.session_state.winner = None
        st.session_state.game_over = False


    state, winner = check_game_over_2(st.session_state.board)
    print(state, winner)


    # print(check_game_over_2(st.session_state.board))

    if st.session_state.next_player == 'O' and state == ONGOING:
        move = get_best_move(st.session_state.board)
        print(move)
        st.session_state.board[move] = 1
        st.session_state.next_player = 'X'

    # check again after agent move 
    won, winner = check_game_over_2(st.session_state.board)

    def handle_click(i):
        if st.session_state.board[i] == 0:
            st.session_state.board[i] = -1
            st.session_state.next_player = 'O'

    # Write rows first 
    for column in range(0, 3):
        cols = st.columns(3)
        for i, row in enumerate(st.session_state.board[column*3:column*3+3]):
            label = '-'
            if row == -1:
                label = 'O'
            elif row == 1:
                label = 'X'

            cols[i].button(
                label,
                key=f"{i}-{column}",
                on_click=handle_click,
                args=(column*3+i,),
            )
        
    if winner != None:
        if winner == -1:
            st.success(f"You won against our bot :party:")
        elif winner == 1:
            st.error(f"Our bot is better than yours :evil: ")

    if state == TIE:
        st.info('That\' a tie. ')


    st.write("""
    #### Project by András Sass, Oto Alves, Lukas Mölschl
    """)


if __name__ == "__main__":
    show()