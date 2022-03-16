import tkinter as tk
import tkinter.ttk as ttk

# Thanks to David Duran (https://stackoverflow.com/a/63345653/1460057) for the idea behind Docker

class Docker:
    def __init__(self, parent, *args, root=None, unDockFunction=None, reDockFunction=None, **kwargs):
        # parent = widget to assign as DockableFrame's parent
        # root = tkinter.Tk widget (the root window widget)
        # unDockFunction = a function to call when undocking - should take one
        #   argument, the Docker object itself.
        # reDockFunction = a function to call when redocking - should take one
        #   argument, the Docker object itself.
#        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        if root is None:
            raise TypeError('root argument must be a tkinter.Tk object')
        self.root = root
        self.docker = tk.Frame(self.parent, *args, **kwargs)
        self.unDockFunction = unDockFunction
        self.reDockFunction = reDockFunction

        self.unDockButton = ttk.Button(self.docker, text="[=]=>", command=self.unDock)
        self.reDockButton = ttk.Button(self.docker, text="[<=]=", command=self.reDock)
        self.isDocked = True

    def unDock(self):
        # Pop widget off as its own window
        if not self.isDocked:
            return
        self.root.wm_manage(self.docker)
        self.isDocked = False
        if self.unDockFunction is not None:
            self.unDockFunction(self)

    def reDock(self):
        # Return widget to a
        if self.isDocked:
            return
        self.root.wm_forget(self.docker)
        self.isDocked = True
        if self.reDockFunction is not None:
            self.reDockFunction(self)

if __name__ == "__main__":
    # Demo

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
