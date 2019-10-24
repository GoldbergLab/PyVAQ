import tkinter as tk
import tkinter.ttk as ttk

class gui():
    def __init__(self, master):
        self.master = master
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure('TEntry', fieldbackground='white')
        self.style.configure('ValidContents.TEntry', fieldbackground='#C1FFC1')
        self.style.configure('InvalidContents.TEntry', fieldbackground='#FFC1C1')
        self.label = ttk.Label(self.master, text="Type \"yay\"")
        self.entry = ttk.Entry(self.master, style='TEntry')
        self.entry.bind('<FocusOut>', self.checkEntryContents)
        self.label.grid(row=0)
        self.entry.grid(row=1)
        print(self.style.layout('TEntry'))

    def checkEntryContents(self, *args):
        print("Checking entry contents")
        if self.entry.get() == "yay":
            # Vald contents
            self.entry['style'] = 'ValidContents.TEntry'
        else:
            self.entry['style'] = 'InvalidContents.TEntry'

root = tk.Tk()
p = gui(root)
root.mainloop()
