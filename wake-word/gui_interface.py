import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import sv_ttk

# window = tk.Tk()
root = tk.Tk()
big_frame = ttk.Frame(root)
big_frame.pack(fill="both", expand=True)
big_frame['padding'] = (20,20,20,20)
sv_ttk.set_theme("dark")


title = ttk.Label(big_frame, text="Wake Word Creator")
title.pack(pady=10)

grid = ttk.Frame(root)
grid.rowconfigure(0, weight=1)
grid.columnconfigure(0, weight=1)
grid['padding'] = (100,10,100,10)
grid.pack(fill="both", expand=True)

training_data = ttk.Label(grid, text="Training Data")
training_data.grid(column=0, row=0, pady=10)

training_data_unsplit_training_data = ttk.Label(grid, text="Unsplit Training Data")
training_data_unsplit_training_data.grid(column=0, row=0, pady=10)

training_data_unsplit_training_data_entry = ttk.Entry(grid, width=30)
training_data_unsplit_training_data_entry.grid(column=0, row=1, pady=10)

def select_file_unsplit_training_data():
    file_path = filedialog.askopenfilename()
    print(file_path)


training_data_unsplit_training_data_button = ttk.Button(grid, text="Select Folder of Unsplit Data", command=select_file_unsplit_training_data)
training_data_unsplit_training_data_button.grid(column=1, row=1, pady=10)

entry = ttk.Entry(grid, width=30)
entry.grid(column=0, row=2, pady=10)



def select_file():
    file_path = filedialog.askopenfilename()
    print(file_path)

button = ttk.Button(grid, text="Select Folder for generating training data", command=select_file)
button.grid(column=1, row=2, pady=10)

root.mainloop()

# window.mainloop()

