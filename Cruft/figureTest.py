import tkinter as tk
from tkinter import Scrollbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class ScrollableWindow:

    def __init__(self, master, fig, **options):

#master.resizable(width=False, height=False)
        master.geometry("%dx%d+0+0" % (800, 500))
        master.focus_set()

        fig_wrapper = tk.Frame(master, width=800, height=fig.get_figheight())
        fig_wrapper.pack(fill=tk.BOTH, expand=True)

        fig_canvas = FigureCanvasTkAgg(fig, master=fig_wrapper)

        scrollbar = Scrollbar(fig_wrapper, orient=tk.VERTICAL, command=fig_canvas.get_tk_widget().yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        fig_canvas.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
        fig_canvas.get_tk_widget().config(yscrollcommand = scrollbar.set, scrollregion=fig_canvas.get_tk_widget().bbox("all"), width=800, height=1000)



n_col, n_row = 3, 11

#fig, axes = plt.subplots(figsize=(n_col,n_row*2), ncols=n_col, nrows=n_row)
fig, axes = plt.subplots(figsize=(7.5, 25), ncols=n_col, nrows=n_row)
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes[i,j].set_xlabel("xlabel")
        axes[i,j].set_ylabel("ylabel")
fig.tight_layout()
showStatsWindow = tk.Tk()
showStatsWindow_ = ScrollableWindow(showStatsWindow, fig)
showStatsWindow.mainloop()
