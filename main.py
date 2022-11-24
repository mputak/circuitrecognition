import torch
import cv2 as cv
import math
import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.filedialog as fd
from data_processing import Process
# TODO: Fix updating for the filepaths in main or make another method to call filepath values in class and call in main.
#       Output of the file_explorer should be fed into the model inference.


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        self.files = ()
        self.geometry("400x400")
        self.title("Circuit Digitizer 3000")

        self.files_chosen = 0
        self.label = tk.Label(self, text=f"{self.files_chosen} files chosen.")
        self.label.place(x=200, y=250, anchor="center")

        self.file_button = tk.Button(self, text="Choose files...", command=self.choose_files)
        self.file_button.place(width=100, height=50, x=200, y=200, anchor="center")

        self.mainloop()

    def choose_files(self):
        self.files = fd.askopenfilenames(title="Choose files.")
        self.file_button.configure(text="Digitize!", command=self.digitize)

        self.files_chosen = len(self.files)
        self.label.config(text=f"{self.files_chosen} files chosen.")

    def digitize(self):
        Process(self.files)

# TODO: Create requirements.txt that user needs to pip install!
#       Create simple GUI for image selection
#           - Allow single and batch images
#           - Drag and drop
#           - Possibly some inference info (time)


MainWindow()
