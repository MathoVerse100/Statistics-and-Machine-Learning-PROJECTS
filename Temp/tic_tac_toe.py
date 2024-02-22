import numpy as np


class none_GUI_multidimensional_Tic_Tac_Toe:
    def __init__(self, dimension):
        self.dimension = dimension
        self.frame = np.array([["_"]*dimension]*dimension)

    def __str__(self):
        return f"This is a {self.dimension} - dimensional Tic Tac Toe game."

    def turn(self):
        if np.count_nonzero(self.frame == 'x') == np.count_nonzero(self.frame == 'o'):
            return 'x'
        else:
            return 'o'

    def anti_turn(self):
        return [i for i in ['x', 'o'] if i != self.turn()][0]

    def win(self):
        win_state = np.array([self.anti_turn()]*self.dimension)
        if np.any([np.all(self.frame[i] == win_state) for i in range(self.dimension)]) or np.any([np.all(
                self.frame.transpose()[i] == win_state) for i in range(self.dimension)]):
            return True
        elif np.all(self.frame.diagonal() == win_state) or np.all(np.fliplr(self.frame).diagonal() == win_state):
            return True
        else:
            return False


def main():
    while True:
        prompt = input("Would you like to play Tic_Tac_Toe? (Y/N) ").lower()
        if prompt != "y" and prompt != "n":
            print("Invalid prompt...")
        else:
            break
    if prompt == "y":
        while True:
            dimension = input("Choose the dimension of the game (2 < n < 10): ")
            if not dimension.isdigit() or int(dimension) <= 2 or int(dimension) >= 10:
                print("Invalid entry...")
            else:
                dimension = int(dimension)
                game = none_GUI_multidimensional_Tic_Tac_Toe(dimension)
                play(game, dimension)
                break
    else:
        print("Goodbye!")


def play(game, dimension):
    frame_coordinates = [[i*10 + j for j in range(1, dimension + 1)] for i in range(1, dimension + 1)]
    admissible_values = [x for sublist in frame_coordinates for x in sublist]
    while True:
        print(game.frame)
        if np.count_nonzero(game.frame == '_') == 0:
            print("Draw!")
            return 0
        if game.win():
            print(f"Game! {game.anti_turn()} wins!")
            return 0
        current_turn = game.turn()
        while True:
            position = input(f"{current_turn} - position: ")
            if not position.isdigit() or (int(position) not in admissible_values):
                print("Invalid entry...")
            else:
                str_position = [int(i) - 1 for i in list(position)]
                if game.frame[str_position[0], str_position[1]] in ['x', 'o']:
                    print("Invalid position...")
                else:
                    game.frame[str_position[0], str_position[1]] = current_turn
                    break


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        exit()
