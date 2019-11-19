import tkinter as tk
from tkinter import Scrollbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class ScrollableWindow:

    def __init__(self, master, fig, **options):
        def on_configure(event):
            # update scrollregion after starting 'mainloop'
            # when all widgets are in canvas
            canvas.configure(scrollregion=canvas.bbox('all'))

            # expand canvas_frame when canvas changes its size
            canvas_width = event.width
            canvas.itemconfig(canvas_frame, width=canvas_width)


        # --- create canvas with scrollbar ---
        canvas = tk.Canvas(master, )
        canvas.pack(side=tk.LEFT, fill='both', expand=True)

        scrollbar = tk.Scrollbar(master, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill='both')

        canvas.configure(yscrollcommand=scrollbar.set)

        # update scrollregion after starting 'mainloop'
        # when all widgets are in canvas
        canvas.bind('<Configure>', on_configure)

        # --- put frame in canvas ---


#        master.resizable(width=False, height=False)

        master.geometry("%dx%d+0+0" % (800, 500))
        master.focus_set()

        fig_wrapper = tk.Frame(canvas)
        canvas_frame= canvas.create_window((0, 0), window=fig_wrapper,)

        fig_canvas = FigureCanvasTkAgg(fig, master=fig_wrapper)
        fig_canvas.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT, expand=True)


n_col, n_row = 3, 11

fig, axes = plt.subplots(figsize=(n_col*2,n_row*2), ncols=n_col, nrows=n_row,)
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes[i,j].set_xlabel("xlabel")
        axes[i,j].set_ylabel("ylabel")
fig.tight_layout()
showStatsWindow = tk.Tk()
showStatsWindow_ = ScrollableWindow(showStatsWindow, fig)
showStatsWindow.mainloop()
