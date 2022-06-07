import tkinter as tk
import tkinter.ttk as ttk

class CollapsableFrame(tk.Frame):
    def __init__(self, parent, *args, collapseSymbol='/\\', collapseText='',
        expandSymbol='\\/', expandText='', collapseFunction=None,
        expandFunction=None, **kwargs):
        # CollapsableFrame: A class that provides a "collapsable" frame, meaning
        #   a Tkinter frame that can be "collapsed" such that all its children
        #   are hidden and the frame becomes shorted, and "expanded" again.
        #   Note that this class is not itself a tkinter widget, but contains
        #   as attributes the widgets that implement the collapsable frame.
        #
        #   Note that the attribute CollapsableFrame.frame is the frame that
        #   actually gets collapsed/expanded - assign CollapsableFrame.frame as
        #   the parent for any widgets that should be in the collapsable frame.
        #
        #
        # parent = container widget to assign as the collapsable frame's parent
        # root = tkinter.Tk widget (the root window widget)
        # collapseFunction = a function to call when collapsing - should take
        #   one argument, the CollapsableFrame object itself.
        # expandFunction = a function to call when expanding - should take one
        #   argument, the CollapsableFrame object itself.
        # collapseSymbol = the symbol to display to indicate the frame may be
        #   collapsed
        # expandSymbol = the symbol to display to indicate the frame may be
        #   expanded.
        # collapseText = the header text for the collapsable frame when it is
        #   in its expanded configuration. This will be displayed as a header,
        #   and will still be visible even when the frame is collapsed.
        # expandText = the header text for the collapsable frame when it is
        #   in its collapsed configuration. This will be displayed as a header,
        #   and will still be visible even when the frame is collapsed.
        #
        super().__init__(parent)
        self.parent = parent
        self.frame = tk.Frame(self, *args, **kwargs)

        # Store user-supplied collapse/expand callbacks
        self.collapseFunction = collapseFunction
        self.expandFunction = expandFunction

        # Store symbols to display to indicate that the frame may be collapsed
        #   or expanded
        self.collapseSymbol = collapseSymbol
        self.expandSymbol = expandSymbol

        # Set boolean isCollapsed flag to True
        self._isCollapsed = False

        # Create collapse/expand button
        self.collapseText = collapseText
        self.expandText = expandText
        self.stateChangeButton = ttk.Button(self)
        self.updateStateChangeButton()

        # Lay out widgets
        self.stateChangeButton.grid(row=0, column=0, sticky=tk.NSEW)
        self.frame.grid(row=1, column=0, sticky=tk.NSEW)

    def updateStateChangeButton(self):
        if self.isCollapsed():
            text = '{s} {t} {s}'.format(s=self.expandSymbol, t=self.expandText)
            fcn = self.expand
        else:
            text = '{s} {t} {s}'.format(s=self.collapseSymbol, t=self.collapseText)
            fcn = self.collapse
        self.stateChangeButton['text'] = text
        self.stateChangeButton['command'] = fcn


    def isCollapsed(self):
        # Return whether frame is in docked or undocked state
        return self._isCollapsed

    def collapse(self):
        # Set boolean isDocked flag to False
        self._isCollapsed = True
        # Update state change button
        self.updateStateChangeButton()
        # Hide frame
        self.frame.grid_forget()
        # If present, run user-supplied undock callbacks
        if self.collapseFunction is not None:
            self.collapseFunction(self.frame)

    def expand(self):
        # Set boolean isDocked flag to False
        self._isCollapsed = False
        # Update state change button
        self.updateStateChangeButton()
        # Show frame
        self.frame.grid()
        # If present, run user-supplied undock callbacks
        if self.expandFunction is not None:
            self.expandFunction(self.frame)

if __name__ == "__main__":
    # CollapsableFrame demo:

    # Create root window
    root = tk.Tk()

    # Create and display an outer frame
    mainFrame = tk.LabelFrame(root, text='Collapsable Frame Demo')
    mainFrame.grid()

    # Create the CollapsableFrame object. Note that any number of keyword
    #   arguments can be passed to format the collapsable frame. See
    #   tkinter.Frame documentation.
    cf = CollapsableFrame(mainFrame, collapseText='Collapse the frame!',
        expandText='Expand the frame!',borderwidth=3, relief=tk.SUNKEN)

    # Create & display a label that is outside the docker frame
    outerLabel = tk.Label(mainFrame, text='That there is a collapsable frame! \/\/\/')
    outerLabel.grid(row=0, column=0)

    # Display the docker frame
    cf.grid(row=1, column=0, sticky=tk.NSEW)
    cf.grid_columnconfigure(0, weight=1)

    # A label that's in the docker frame. Notice that the parent is
    #   the Docker.docker object, which is a tkinter.Frame
    innerLabel1 = tk.Label(cf.frame, text='Hi there, I am inside a collapsable frame!')
    innerLabel2 = tk.Label(cf.frame, text='Me too!')

    # Display inner label
    innerLabel1.grid(row=1, column=0)
    innerLabel2.grid(row=2, column=0)

    # Start main GUI loop
    root.mainloop()
