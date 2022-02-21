import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from itertools import product
import random
import pickle

"""
The global variables BOARD_SIZE and STATES are declared:
 STATES is a list of all possible board positions. 
 Each board position is a state and saved as a 1-d tuple of length 9 
 such that the first 3 entries are the first row of the field; entries 4-6 are the second row, ...
 1 corresponds to a cross, -1 to an O and 0 to an empty field.
"""

BOARD_SIZE = 9
choices = [[-1, 0, 1]] * BOARD_SIZE
all_states = list(product(*choices))
STATES = [state for state in all_states if abs(sum(list(state))) <= 1]
EXT_CTR = 0
# print(STATES)


class Player:
    """
    The Player class has an assigned value function,
    some parameters which influence learning of a player
    and methods based to the value function.
    """
    def __init__(self,
                 value_function="random",
                 alpha=0.8,
                 alpha_decay=0,
                 epsilon=0.1,
                 epsilon_decay=0,
                 initial_value=1):
        """
        :param value_function: A specific value function can be loaded
        by specifying its name. Alternatively "random" can be specified
        to initialize a random value function.
        :param alpha: Determines the (initial) learning rate
        :param alpha_decay: If 0, then the learning rate stays constant during training.
        If n != 0, then the learning rate decays exponentially such that
        after n training games the learning rate is exp(-10).
        :param epsilon: Determines the (initial) probability to select a random move (exploration)
        :param epsilon_decay: Similar to alpha_decay
        """

        self.num_games = 0
        self.update_ctr = 0
        self.sign = 1
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.c = self.alpha_decay ** 2 / 10
        self.learning_rate = self.alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_greedy = epsilon
        if value_function == "random":
            self.value_function = initialize_random_value_function(initial_value=initial_value)
        else:
            path = "value_functions/" + value_function + "_value_function.pkl"
            file = open(path, "rb")
            self.value_function = pickle.load(file)
            file.close()

    def choose_move(self, state):
        """
        For a given state a move is returned according to the
        value function using epsilon greedy strategy.
        :param state: The state for which a move should be determined
        :return: the state after the move
        """
        indexes_open_fields = [i for i, j in enumerate(state) if j == 0]
        available_states = []
        max_value = -10
        chosen_state = None
        for i in indexes_open_fields:
            available_state = state.copy()
            available_state[i] = self.sign
            available_state = tuple(available_state)
            available_states.append(available_state)
            state_value = self.value_function[available_state]
            if state_value >= max_value:
                max_value = state_value
                chosen_state = available_state
        if self.epsilon_decay != 0:
            self.epsilon_greedy = self.epsilon * np.exp(- self.num_games ** 2 / self.c)  # Gaussian s.t. at game number epsilon_decay it is exp(-10)
        if self.epsilon_greedy > np.random.uniform(0, 1):
            chosen_state = random.choice(available_states)
        return list(chosen_state)

    def update_value_function(self, state, next_state, game_over=False, winner=None):
        """
        Using the update: V(S_t) = V(S_t) + learning_rate * (V(S_t+1) - V(S_t))
        with S_t being the current state and S_t+1 the next state,
        the value function of the player is updated
        :param state: current state S_t
        :param next_state: next state S_t+1
        """
        if self.alpha_decay != 0:
            self.learning_rate = np.exp(
                - self.num_games ** 2 / self.c)  # Gaussian s.t. at game number alpha_decay it is exp(-10)
        s_t = self.value_function[state]
        s_t_1 = self.value_function[next_state]
        self.value_function[state] = s_t + self.learning_rate * (s_t_1 - s_t)

    def save_value_function(self, name=""):
        """
        Saves the players value function as a pickle file.
        :param name: (Part of the) name of the resulting file
        """
        file = open("value_functions/" + name + "_value_function.pkl", "wb")
        pickle.dump(self.value_function, file)
        file.close()


class PlayerMC(Player):

    def __init__(self, value_function="random", alpha=0.8, alpha_decay=0,
                 epsilon=0.1, epsilon_decay=0, gamma=1):
        super().__init__(value_function, alpha, alpha_decay,
                 epsilon, epsilon_decay)
        self.return_lists = initialize_return_lists()
        self.return_means = initialize_return_means()
        self.visited_states_episode = []
        self.gamma = gamma
        self.counts = initialize_Ns()

    def update_value_function(self, state, next_state, game_over=False, winner=None):
        """
        Using the update: V(S_t) = Average(Returns(S_t))
        with S_t being the current state and Returns being
        the list of Returns for the state over the episodes
        :param state: current state S_t
        :param next_state: next state S_t+1
        """
        if self.alpha_decay != 0:
            self.learning_rate = np.exp(
                - self.num_games ** 2 / self.c)  # Gaussian s.t. at game number alpha_decay it is exp(-10)

        self.visited_states_episode.append(state)
        print('not yet')
        if check_game_over(state):
            g = 0
            for state_idx, state in enumerate(self.visited_states_episode[0:-1]):
                next_state = self.visited_states_episode[state_idx+1]
                v_t = self.value_function[state]
                v_t_1 = self.value_function[next_state]
                g = self.gamma * g + (v_t_1 - v_t)
                self.return_lists[state].append(g)
                self.value_function[state] = np.mean(self.return_lists[state])

            self.visited_states = []  # resetting visited states for next episode


class PlayerMCEfficient(PlayerMC):

    def update_value_function(self, state, next_state, game_over=False, winner=None):
        """
        Using the update: V(S_t) = Average(Returns(S_t))
        with S_t being the current state and Returns being
        the list of Returns for the state over the episodes
        :param state: current state S_t
        :param next_state: next state S_t+1
        """
        if self.alpha_decay != 0:
            self.learning_rate = np.exp(
                - self.num_games ** 2 / self.c)  # Gaussian s.t. at game number alpha_decay it is exp(-10)

        self.visited_states_episode.append(next_state)

        if game_over:
            g = 0
            if self.alpha > 0: #micro optimization
                for state_idx, state in enumerate(self.visited_states_episode[0:-1]):
                    next_state = self.visited_states_episode[state_idx + 1]
                    v_t = self.value_function[state]
                    v_t_1 = self.value_function[next_state]
                    g = self.gamma * g + (v_t_1 - v_t)

                    # self.counts[state] +=1
                    # self.value_function[state] = self.value_function[state] + (1/(self.counts[state])) *
                    # (g - self.value_function[state])
                    self.value_function[state] = self.value_function[state] + self.learning_rate / 2 * (
                            g - self.value_function[state])

            self.visited_states_episode = []  # resetting visited states for next episode


from minimax import get_best_move


class PlayerMinmax(Player):
    def __init__(self, value_function="random", alpha=0.8, alpha_decay=0,
                epsilon=0.1, epsilon_decay=0, gamma=1):
        # Init the class anyway so the has the `self.sign` property
        super().__init__(value_function, alpha, alpha_decay,
                epsilon, epsilon_decay)

    def choose_move(self, state):
        """
        For a given state a move is returned according to the
        value function using the minimax algorithm.
        :param state: The state for which a move should be determined
        :return: the state after the move
        """
        move_position = get_best_move(state, player_sign=self.sign)
        state[move_position] = self.sign
        return state

    def update_value_function(self, state, next_state, game_over=False, winner=None):
        pass


class Dojo:
    """
    In a dojo 2 players a and b are present. The players can play against each other
    to train (update their value function). Additionally a score is being kept
    from which statistics about the players performances can be calculated/plotted.
    """
    def __init__(self,
                 value_function_a="random",
                 value_function_b="random",
                 alpha_a=0.8,
                 alpha_b=0.8,
                 alpha_decay_a=0,
                 alpha_decay_b=0,
                 epsilon_a=0.1,
                 epsilon_b=0.1,
                 epsilon_decay_a=0,
                 epsilon_decay_b=0,
                 algo='TD',
                 algo2='TD',
                 gamma=1):
        """
        2 players a and b are instantiated and the the attributes for keeping score are initialized
        :param value_function_a: A specific value function can be loaded
        by specifying its name. Alternatively "random" can be specified
        to initialize a random value function.
        :param value_function_b: same
        :param alpha_a: Determines the (initial) learning rate for a
        :param alpha_b: Determines the (initial) learning rate for b
        :param alpha_decay_a: If 0, then the learning rate stays constant during training.
        If > 0, then the learning rate decays exponentially such that
        after alpha_decay_a training games the learning rate is exp(-10)
        :param alpha_decay_b: If 0, then the learning rate stays constant during training.
        If > 0, then the learning rate decays exponentially such that
        after alpha_decay_b training games the learning rate is exp(-10)
        :param epsilon_a: Determines the (initial) probability to select a random move (exploration)
        :param epsilon_b: Determines the (initial) probability to select a random move (exploration)
        :param epsilon_decay_a: Similar to alpha_decay
        :param epsilon_decay_b: Similar to alpha_decay
        """

        if algo == 'TD':
            self.player_a = Player(value_function_a, alpha_a, alpha_decay_a, epsilon_a, epsilon_decay_a)
        elif algo == 'MC':
            self.player_a = PlayerMCEfficient(value_function_a, alpha_a, alpha_decay_a, epsilon_a, epsilon_decay_a, gamma)
        elif algo == 'minimax':
            self.player_a = PlayerMinmax()

        if algo2 == 'TD':
            self.player_b = Player(value_function_b, alpha_b, alpha_decay_b, epsilon_b, epsilon_decay_b)
        elif algo2 == 'MC':
            self.player_b = PlayerMCEfficient(value_function_b, alpha_b, alpha_decay_b, epsilon_b, epsilon_decay_b, gamma)
        elif algo2 == 'minimax':
            self.player_b = PlayerMinmax()

    

        self.wins = []
        self.stats = pd.DataFrame(columns=["wins_a", "wins_b", "cum_wins_a", "cum_wins_b", "winrate_a", "winrate_b"])

    def play(self, num_games=100):
        """
        Player a and b play a specified number of games and
        their value functions are updated during the games
        :param num_games: The number of games to play (train)
        """

        # Every player gets a symbol (X=1, O=-1)
        self.player_a.sign = 1
        self.player_b.sign = -1

        for game_num in range(1, num_games + 1):
            print(f"game {game_num}/{num_games}", end="\r")
            board = [0] * BOARD_SIZE                    # Board is initialized
            boards = [tuple(board)]
            self.player_a.sign *= -1                    # after every game the symbols are switched
            self.player_b.sign *= -1
            a_begins = 1 if game_num % 2 == 1 else 0
            move_counter = 1
            game_over = False
            # Loop over all the moves of a game
            while not game_over:
                player = self.player_a if move_counter % 2 == a_begins else self.player_b   # player a or b in alternating order
                board = player.choose_move(board)
                boards.append(tuple(board))
                game_over, winner = check_game_over(board)
                if move_counter >= 2:
                    # The previous and current afterstates are needed (the state in between is part of the environment)
                    player.update_value_function(state=boards[-3], next_state=boards[-1], game_over=(game_over & (winner!=0)), winner=winner)
                if game_over:
                    if winner == 0:
                        boards.append("draw_state")
                        player.update_value_function(state=boards[-2], next_state=boards[-1], game_over=game_over, winner=winner)
                    else:
                        boards.append("loser_state")
                    player_2 = self.player_b if move_counter % 2 == a_begins else self.player_a
                    player_2.update_value_function(state=boards[-3], next_state=boards[-1], game_over=game_over, winner=winner)
                    player.num_games += 1
                    player_2.num_games += 1
                    if winner == 1:
                        if self.player_a.sign == 1:
                            self.wins.append("a")
                        elif self.player_b.sign == 1:
                            self.wins.append("b")
                    elif winner == -1:
                        if self.player_a.sign == -1:
                            self.wins.append("a")
                        elif self.player_b.sign == -1:
                            self.wins.append("b")
                    elif winner == 0:
                        self.wins.append("draw")
                move_counter += 1

    def calculate_stats(self, moving_average=1000, show_plot=True):
        """
        From the counted wins the winrates of player a and b are calculated.
        A moving average is used to visualize the evolution of the winrates in a smooth manner.
        :param moving_average: How many previous games should be used to calculate the average
        :param show_plot: Boolean
        """
        self.stats["wins_a"] = pd.Series([1 if win == "a" else 0 for win in self.wins])
        self.stats["wins_b"] = pd.Series([1 if win == "b" else 0 for win in self.wins])
        self.stats["cum_wins_a"] = self.stats["wins_a"].cumsum()
        self.stats["cum_wins_b"] = self.stats["wins_b"].cumsum()
        self.stats["winrate_a"] = self.stats["wins_a"].rolling(moving_average).sum() / moving_average
        self.stats["winrate_b"] = self.stats["wins_b"].rolling(moving_average).sum() / moving_average
        if show_plot:
            fig, ax = plt.subplots()
            ax.plot(self.stats["winrate_a"], label='player a')
            ax.plot(self.stats["winrate_b"], label='player b')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
            ax.set_xlabel("games")
            ax.set_ylabel("winrate")
            ax.legend()
            plt.show()


class Competition(Dojo):
    """
    In a competition 2 players play games against each other with the aim to outperform the other and not to learn.
    No updates to the value functions of the 2 players are made and no random actions are taken.
    A dojo is meant for training (value functions are updated and sometimes random actions
    are taken for exploration sake), but during a competition the strategies of the 2 players are evaluated.
    """
    def __init__(self, value_function_a="random", value_function_b="random", algo='TD', algo2='TD', gamma=1):
        """
        Initialize the 2 players with 0 learning rates and no chance of selecting random actions
        :param value_function_a: random or loaded by name
        :param value_function_b: random or loaded by name
        """

        super().__init__(value_function_a,
                         value_function_b,
                         alpha_a=0,
                         alpha_b=0,
                         alpha_decay_a=0,
                         alpha_decay_b=0,
                         epsilon_a=0,
                         epsilon_b=0,
                         epsilon_decay_a=0,
                         epsilon_decay_b=0,
                         algo=algo,
                         algo2=algo2,
                         gamma=gamma
                         )


def initialize_random_value_function(initial_value=1):
    """
    The value function is a dictionary with every possible board as the keys and
    the associated value of that board as the value. Here a random value function
    is initialized.
    Winning boards get the value 1
    The loser_state gets the value -1
    The draw_state gets the value 0
    Every other state gets a value between 1 and 1.1 (exploring starts)
    :return: The value function as a dictionary
    """
    values = initial_value + np.random.uniform(0, 0.01, len(STATES))
    state_values = zip(STATES, values)
    value_function = dict(state_values)
    for state in value_function:
        board = np.array(state).reshape((3, 3))
        winning_positions = list(abs(board.sum(0))) + list(abs(board.sum(1))) + [abs(board.trace(0))] + [
            abs(board.trace(1))]
        if 3 in winning_positions:
            value_function[state] = 1
    value_function["loser_state"] = -1
    value_function["draw_state"] = 0
    return value_function


def initialize_return_lists():
    return_lists = dict(zip(STATES, [[] for i in range(len(STATES))]))
    # print(return_lists)
    return return_lists


def initialize_return_means():
    return_means = dict(zip(STATES, [(None,0) for i in range(len(STATES))]))
    # print(return_lists)
    return return_means


def initialize_Ns():
    Ns = dict(zip(STATES, [0 for i in range(len(STATES))]))
    # print(return_lists)
    return Ns


def check_game_over(board):
    """
    Check if a given board is the end of the game (winning position or draw)
    and return a boolean and who won (1: X, -1: O, 0: draw, None: not the end of the game)
    :param board: The board to evaluate
    :return: boolean and who won
    """
    board = np.array(board).reshape((3, 3))
    winning_positions = list(board.sum(0)) + list(board.sum(1)) + [board.trace(0)] + [board.trace(1)]
    game_over = False
    winner = None
    if 3 in winning_positions:
        game_over = True
        winner = 1
    elif -3 in winning_positions:
        game_over = True
        winner = -1
    elif 0 not in board:
        game_over = True
        winner = 0
    return game_over, winner


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


def train_test_session(num_train_games=50000,
                       num_test_games=5000,
                       value_function_a="random",
                       value_function_b="random",
                       alpha_a=1,
                       alpha_b=1,
                       alpha_decay_a=50000,
                       alpha_decay_b=50000,
                       epsilon_a=0.1,
                       epsilon_b=0.1,
                       epsilon_decay_a=50000,
                       epsilon_decay_b=50000,
                       algo='TD',
                       algo2=None,
                       gamma=1,
                       name_value_function_a="a",
                       name_value_function_b="b",
                       moving_average=1000,
                       save=True,
                       show_plot=True
                       ):
    """
    A dojo with 2 players is initialized and a set number of training
    games are played before the resulting value functions are saved.
    To test the (or any other previously saved) saved value functions
    a competition is initialized to test the 2 value functions against
    each other.
    :param num_train_games: number of games for training (dojo)
    :param num_test_games: number of games for testing (competition)
    :param value_function_a: random or a specific value function which should be used for training
    :param value_function_b: random or a specific value function which should be used for training
    :param alpha_a: Determines the (initial) learning rate for a
    :param alpha_b: Determines the (initial) learning rate for b
    :param alpha_decay_a: 0: no decay, >0: exponential decay s.t.
    after alpha_decay_a games the learning rate is alpha_a * exp(-10)
    :param alpha_decay_b: 0: no decay, >0: exponential decay s.t.
    after alpha_decay_b games the learning rate is alpha_b * exp(-10)
    :param epsilon_a: Determines the (initial) probability to select a random move (exploration)
    :param epsilon_b: Determines the (initial) probability to select b random move (exploration)
    :param epsilon_decay_a: same as alpha_decay_a
    :param epsilon_decay_b: same as to alpha_decay_b
    :param name_value_function_a: name under which the value function of a should be saved
    :param name_value_function_b: name under which the value function of b should be saved
    :param moving_average: Number of games used to calculate moving average of winning rate
    :param save: boolean
    :param show_plot: boolean
    :return: returns the training and testing stats
    """

    if algo2 == None:
        algo2 = algo

    dojo = Dojo(value_function_a=value_function_a,
                value_function_b=value_function_b,
                alpha_a=alpha_a,
                alpha_b=alpha_b,
                alpha_decay_a=alpha_decay_a,
                alpha_decay_b=alpha_decay_b,
                epsilon_a=epsilon_a,
                epsilon_b=epsilon_b,
                epsilon_decay_a=epsilon_decay_a,
                epsilon_decay_b=epsilon_decay_b,
                algo=algo,
                algo2=algo2,
                gamma=gamma
                )
    if num_train_games > 0:
        print("Training:")
        dojo.play(num_train_games)
        dojo.calculate_stats(moving_average=moving_average, show_plot=show_plot)
    if save:
        dojo.player_a.save_value_function(name=name_value_function_a)
        dojo.player_b.save_value_function(name=name_value_function_b)

    competition = Competition(algo=algo, algo2=algo2, gamma=gamma)
    competition.player_a.value_function = dojo.player_a.value_function
    competition.player_b.value_function = dojo.player_b.value_function

    if num_test_games > 0:
        print("\nTesting:")
        competition.play(num_test_games)
        competition.calculate_stats(moving_average=moving_average, show_plot=show_plot)

    return dojo.stats, competition.stats


def average_train_test_sessions(num_experiments=5,
                                num_train_games=50000,
                                num_test_games=5000,
                                value_function_a="random",
                                value_function_b="random",
                                alpha_a=1,
                                alpha_b=1,
                                alpha_decay_a=50000,
                                alpha_decay_b=50000,
                                epsilon_a=0.1,
                                epsilon_b=0.1,
                                algo='TD',
                                gamma=1,
                                epsilon_decay_a=50000,
                                epsilon_decay_b=50000,
                                moving_average=1000,
                                show_plot=True
                                ):
    """
    Makes number of training sessions and averages the resulting training and testing results.
    :param num_experiments: number of train and test sessions
    :param num_train_games: number of train games per session
    :param num_test_games: number of test games per session
    :param value_function_a: random or a specific value function which should be used for training
    :param value_function_b: random or a specific value function which should be used for training
    :param alpha_a: Determines the (initial) learning rate for a
    :param alpha_b: Determines the (initial) learning rate for b
    :param alpha_decay_a: 0: no decay, >0: exponential decay s.t.
    after alpha_decay_a games the learning rate is alpha_a * exp(-10)
    :param alpha_decay_b: 0: no decay, >0: exponential decay s.t.
    after alpha_decay_b games the learning rate is alpha_b * exp(-10)
    :param epsilon_a: Determines the (initial) probability to select a random move (exploration)
    :param epsilon_b: Determines the (initial) probability to select b random move (exploration)
    :param epsilon_decay_a: same as alpha_decay_a
    :param epsilon_decay_b: same as to alpha_decay_b
    :param moving_average: Number of games used to calculate moving average of winning rate
    :param show_plot: boolean
    :return: All resulting stats and the average stats in a dictionary
    """

    results = {"train": {"winrate_a": pd.DataFrame(), "winrate_b": pd.DataFrame()},
               "test": {"winrate_a": pd.DataFrame(), "winrate_b": pd.DataFrame()}}

    for experiment in range(num_experiments):
        print(f"\nExperiment {experiment + 1}/{num_experiments}")
        train_stats, test_stats = train_test_session(num_train_games=num_train_games,
                                                     num_test_games=num_test_games,
                                                     value_function_a=value_function_a,
                                                     value_function_b=value_function_b,
                                                     alpha_a=alpha_a,
                                                     alpha_b=alpha_b,
                                                     alpha_decay_a=alpha_decay_a,
                                                     alpha_decay_b=alpha_decay_b,
                                                     epsilon_a=epsilon_a,
                                                     epsilon_b=epsilon_b,
                                                     epsilon_decay_a=epsilon_decay_a,
                                                     epsilon_decay_b=epsilon_decay_b,
                                                     algo=algo,
                                                     gamma=gamma,
                                                     moving_average=moving_average,
                                                     save=False,
                                                     show_plot=False
                                                     )
        results["train"]["winrate_a"][experiment] = train_stats["winrate_a"]
        results["train"]["winrate_b"][experiment] = train_stats["winrate_b"]
        results["test"]["winrate_a"][experiment] = test_stats["winrate_a"]
        results["test"]["winrate_b"][experiment] = test_stats["winrate_b"]

    results["train"]["winrate_a"]['avg_train_winrate_a'] = results["train"]["winrate_a"].mean(axis=1)
    results["train"]["winrate_b"]['avg_train_winrate_b'] = results["train"]["winrate_b"].mean(axis=1)
    results["test"]["winrate_a"]['avg_test_winrate_a'] = results["test"]["winrate_a"].mean(axis=1)
    results["test"]["winrate_b"]['avg_test_winrate_b'] = results["test"]["winrate_b"].mean(axis=1)

    if show_plot:
        if num_train_games > 0:
            fig, ax = plt.subplots()
            ax.plot(results["train"]["winrate_a"]['avg_train_winrate_a'], label='player a')
            ax.plot(results["train"]["winrate_b"]['avg_train_winrate_b'], label='player b')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
            ax.set_xlabel("games")
            ax.set_ylabel("winrate")
            ax.legend()
            plt.show()
        if num_test_games > 0:
            fig, ax = plt.subplots()
            ax.plot(results["test"]["winrate_a"]['avg_test_winrate_a'], label='player a')
            ax.plot(results["test"]["winrate_b"]['avg_test_winrate_b'], label='player b')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
            ax.set_xlabel("games")
            ax.set_ylabel("winrate")
            ax.legend()
            plt.show()

    return results

