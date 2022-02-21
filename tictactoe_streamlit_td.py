import streamlit as st
import numpy as np
from utils import Player
from minimax import check_game_over_2


RL_agent = Player('a_optimal_TD', 0., 0., 0., 0.)
RL_agent.sign = 1

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
        #### Interactively play against our RL Agent 
        This agent was trained using one-step Temporal Difference learning
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
        st.session_state.board = RL_agent.choose_move(st.session_state.board)
        st.session_state.next_player = 'X'

    # check again after agent move 
    won, winner = check_game_over_2(st.session_state.board)

    def handle_click(i):
        if st.session_state.board[i] == 0:
            st.session_state.board[i] = -1
            st.session_state.next_player = 'O'
            print(st.session_state.board)

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
        st.info('That\'s a tie. ')


    st.write("""
    #### Project by András Sass, Oto Alves, Lukas Mölschl
    """)


if __name__ == "__main__":
    show()