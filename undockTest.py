#from MonitorWidgets import AudioMonitor as am
from DockableFrame import Docker
import tkinter as tk
root = tk.Tk()

def unDockFunction(d):
    pass

def reDockFunction(d):
    d.docker.grid(row=1, column=0)

df = Docker(root, unDockFunction=unDockFunction, reDockFunction=reDockFunction)
labOut = tk.Label(root, text='That there is a dockable frame')
labOut.grid(row=0, column=0)
df.docker.grid(row=1, column=0)
labIn = tk.Label(df.docker, text='hi there I am in a dockable frame.')
df.unDockButton.grid(row=0, column=0)
df.reDockButton.grid(row=0, column=1)
labIn.grid(row=1, column=0, columnspan=2)
root.mainloop()


# import tkinter as tk
#
# root = tk.Tk()
#
# class MyFigure(tk.Frame):
#     def __init__(self, master):
#         tk.Frame.__init__(self,master)
#         self.master = master
#         self.bc = tk.Button(self, text='confi',
#                             command=lambda:self.configure(bg='red')
#                             )
#         self.bmanage = tk.Button(self, text='manage',
#                                  command = lambda:self._manage()
#                                  )
#         self.bforget = tk.Button(self, text='forget',
#                                  command = lambda:self._forget()
#                                  )
#
#         self.bmanage.pack(side='left')
#         self.bc.pack(side='left')
#         self.bforget.pack(side='left')
#         self.frame = tk.Frame(self.master, bg="red", height=100)
#         self.label=tk.Label(self.frame, text="hi")
#         self.frame.pack()
#         self.label.pack(expand=True, fill=tk.BOTH)
#
#     def _manage(self):
#         test=self.master.wm_manage(self.frame)
#
#     def _forget(self):
#         self.master.wm_forget(self.frame)
#         self.frame.pack()
#
# mf = MyFigure(root)
# mf.pack()
# root.mainloop()
