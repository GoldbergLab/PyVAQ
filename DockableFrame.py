import tkinter as tk
import tkinter.ttk as ttk

# Thanks to David Duran (https://stackoverflow.com/a/63345653/1460057) for the
#   implementation idea behind this class

class Docker:
    def __init__(self, parent, *args, root=None, unDockText='[=]=>', labelText='',
        reDockText='[<=]=', unDockFunction=None, reDockFunction=None, **kwargs):
        # Docker: A class that provides a "dockable" frame, meaning a Tkinter
        #   frame that can be "undocked" from the main window, and "redocked"
        #   again, along with buttons for docking and undocking. Note that this
        #   class is not itself a tkinter widget, but contains as attributes
        #   the widgets that implement the dockable frame.
        #
        #   Note that the attribute Docker.docker is the frame that actually
        #   gets docked/undocked - assign Docker.docker as the parent for any
        #   widgets that should be in the dockable frame.
        #
        #   The attributes Docker.unDockButton and Docker.reDockButton are the
        #   buttons that control the docking/undocking. They have the
        #   Docker.docker as their parent widget, and their display behavior
        #   when being docked/undocked can be controlled by passing in a
        #   unDockFunction and/or reDockFunction as arguments.
        #
        # parent = tkinter container widget to assign as the dockable frame's
        #   parent (for when the frame is docked)
        # root = tkinter.Tk widget (the root window widget)
        # unDockFunction = a function to call when undocking - should take one
        #   argument, the Docker object itself.
        # reDockFunction = a function to call when redocking - should take one
        #   argument, the Docker object itself.
        # unDockText = the text to display on the undock button
        # reDockText = the text to display on the redock button
        # labelText = if empty or None, Docker.docker object will be a
        #   tkinter.Frame with no label. If it is a non-empty string, then
        #   Docker.docker will be a tkinter.LabelFrame with the label as given.
        #
        self.parent = parent
        if root is None:
            raise TypeError('root argument must be a tkinter.Tk object')
        self.root = root
        if labelText is None or len(labelText) == 0:
            self.docker = tk.Frame(self.parent, *args, **kwargs)
        else:
            self.docker = tk.LabelFrame(self.parent, *args, text=labelText, **kwargs)

        # Store user-supplied dock/undock callbacks
        self.unDockFunction = unDockFunction
        self.reDockFunction = reDockFunction

        # Create undock and dock buttons
        self.unDockButton = ttk.Button(self.docker, text=unDockText, command=self.unDock)
        self.reDockButton = ttk.Button(self.docker, text=reDockText, command=self.reDock)

        # Set boolean isDocked flag to True
        self._isDocked = True

    def isDocked(self):
        # Return whether frame is in docked or undocked state
        return self._isDocked

    def unDock(self):
        # Pop widget off as its own window
        self.root.wm_manage(self.docker)
        # Set boolean isDocked flag to False
        self._isDocked = False
        # Make the window close "X" button re-dock frame instead of closing it
        tk.Wm.protocol(self.docker, "WM_DELETE_WINDOW", self.reDock)
        # If present, run user-supplied undock callbacks
        if self.unDockFunction is not None:
            self.unDockFunction(self)

    def reDock(self):
        # Return widget to a docked state
        self.root.wm_forget(self.docker)
        # Set boolean isDocked flag to True
        self._isDocked = True
        # If present, run user-supplied redock callbacks
        if self.reDockFunction is not None:
            self.reDockFunction(self)

if __name__ == "__main__":
    # DockableFrame demo:

    # Create root window
    root = tk.Tk()

    # Define a function to be run when Docker is undocking
    def unDockFunction(d):
        # Hide undock button
        d.unDockButton.grid_forget()
        # Show dock button
        d.reDockButton.grid(row=1, column=1)
    # Define a function to be run when Docker is docking
    def reDockFunction(d):
        # Hide dock button
        d.reDockButton.grid_forget()
        # Show undock button
        d.unDockButton.grid(row=1, column=1)
        # Re-display dock button
        d.docker.grid(row=1, column=0)

    # Create and display an outer frame
    mainFrame = tk.LabelFrame(root, text='Dockable Frame demo')
    mainFrame.grid()

    # Create the Docker object. In this case, the main frame is the parent of
    #   the docker frame (it could be any container widget including the root
    #   window). The docker also needs access to the root window, which is the
    #   root keyword argument. Note that any number of keyword arguments can
    #   be passed to format the docker frame. See tkinter.Frame documentation.
    df = Docker(mainFrame, root=root, unDockFunction=unDockFunction,
        reDockFunction=reDockFunction, unDockText='Undock it!',
        reDockText='Dock it!', borderwidth=3, relief=tk.SUNKEN)

    # Create & display a label that is outside the docker frame
    outerLabel = tk.Label(mainFrame, text='That there is a dockable frame! \/\/\/')
    outerLabel.grid(row=0, column=0)

    # Display the docker frame
    df.docker.grid(row=1, column=0)

    # A label that's in the docker frame. Notice that the parent is
    #   the Docker.docker object, which is a tkinter.Frame
    innerLabel = tk.Label(df.docker, text='Hi there, I am inside a dockable frame!')

    # Display inner label
    innerLabel.grid(row=1, column=0)

    # Calling "reDock" properly initializes the docker to the docked state.
    df.reDock()

    # Start main GUI loop
    root.mainloop()
