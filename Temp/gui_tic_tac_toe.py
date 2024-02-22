import numpy as np
from tkinter import *


board = np.array([["_"]*3]*3)
click_counter = 0


def main():
    initial_root = Tk()
    initial_root.title("Dashboard")

    dashboard(initial_root)

    initial_root.mainloop()


def dashboard(initial_root):
    prompt = Label(initial_root, text="Would you like to play Tic_Tac_Toe? (Y/N)", font=("Arial", 26))
    prompt.pack()
    prompt_entry = Entry(initial_root, width=100, justify="center", font=("Arial", 14))
    prompt_entry.bind("<Return>", lambda event: enter(prompt_entry, initial_root))
    prompt_entry.pack()


def enter(prompt_entry, initial_root):
    user_input = prompt_entry.get().lower()
    if user_input not in ["y", "n", "yes", "no"]:
        prompt_entry.config(state=DISABLED)
        new_label = Label(initial_root, text="Invalid prompt. Please retry...", font=("Arial", 12))
        new_label.pack()
        dashboard(initial_root)
    elif user_input in ["n", "no"]:
        def terminate():
            initial_root.destroy()
        new_label = Label(initial_root, text="See you next time! Terminating in 3, 2, 1...", font=("Arial", 12))
        new_label.pack(side="bottom")
        initial_root.after(4000, terminate)
    else:
        Tic_Tac_Toe(initial_root)


def Tic_Tac_Toe(initial_root):
    initial_root.destroy()
    game_root = Tk()
    game_root.title("Tic Tac Toe")

    def myClick(j):
        global click_counter
        global board

        new_frame = Frame(game_root, width=20, height=8, border=4, borderwidth=12,
                          highlightbackground="black", highlightthickness=5, bg="black")
        new_frame.grid(row=(j // 3) % 3, column=j % 3)

        label = Label(new_frame, text="X" if click_counter % 2 == 0 else "O", font=("Arial", 50),
                      fg="white", bg="black")
        label.pack(expand=True, fill="both")
        board[(j // 3) % 3, j % 3] = "X" if click_counter % 2 == 0 else "O"
        if game_state(board) != 0:
            for button in buttons:
                button.config(state=DISABLED)
            final_frame = Frame(game_root)
            final_frame.grid(row=4, columnspan=3)

            final_label = Label(final_frame, text=f"{game_state(board)} Game will close in 5, 4, 3, 2, 1....",
                                font=("Arial", 16), fg="yellow", bg="black")
            final_label.pack()
            game_root.after(5000, terminate_program)
        
        click_counter += 1

    def terminate_program():
        game_root.destroy()

    buttons = [Button(game_root, width=20, height=8, border=4, borderwidth=12, highlightbackground="white",
                      highlightthickness=5, bg="black", command=lambda i=i: myClick(i)) for i in range(1, 10)]

    j = 1
    for button in buttons:
        button.grid(row=(j // 3) % 3, column=j % 3)
        j += 1
    
    game_root.mainloop()


def game_state(board):
    if np.count_nonzero(board == 'X') == np.count_nonzero(board == 'O'):
        win_value = 'O'
    else:
        win_value = 'X'
    
    if np.count_nonzero(board == "_") == 0:
        return "Draw!"
    
    win_state = np.array([win_value]*3)
    if check_horizontal_lines(board, win_state) or check_vertical_lines(board, win_state) or check_diagonal_lines(board, win_state):
        return f"Game! {win_value} is the winner!"
    return 0


def check_horizontal_lines(board, win_state):
    return np.any([np.all(board[i] == win_state) for i in range(3)])


def check_vertical_lines(board, win_state):
    return np.any([np.all(board.transpose()[i] == win_state) for i in range(3)])


def check_diagonal_lines(board, win_state):
    return np.all(board.diagonal() == win_state) or np.all(np.fliplr(board).diagonal() == win_state)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        exit()
