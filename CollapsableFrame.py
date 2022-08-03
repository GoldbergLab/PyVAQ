import tkinter as tk
import tkinter.ttk as ttk

class CollapsableFrame(tk.Frame):
    def __init__(self, parent, *args, collapseSymbol='/\\', collapseText=None,
        expandSymbol='\\/', expandText=None, collapseFunction=None,
        expandFunction=None, collapsed=True, **kwargs):
        # CollapsableFrame: A class that provides a "collapsable" frame, meaning
        #   a Tkinter frame that can be "collapsed" such that all its children
        #   are hidden and the frame becomes shorted, and "expanded" again.
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
        self.parent = parent
        self.outerFrame = tk.Frame(self.parent, *args, **kwargs)
        super().__init__(self.outerFrame)

        # Store user-supplied collapse/expand callbacks
        self.collapseFunction = collapseFunction
        self.expandFunction = expandFunction

        # Store symbols to display to indicate that the frame may be collapsed
        #   or expanded
        self.collapseSymbol = collapseSymbol
        self.expandSymbol = expandSymbol

        # Store text to display on button when frame is collapsed or expanded
        if collapseText is None and expandText is None:
            collapseText = ''
            expandText = ''
        elif collapseText is None and expandText is not None:
            collapseText = expandText
        elif collapseText is not None and expandText is None:
            expandText = collapseText
        self.collapseText = collapseText
        self.expandText = expandText

        # Set boolean isCollapsed flag to True
        self._isCollapsed = collapsed

        # Create collapse/expand button
        self.stateChangeButton = tk.Button(self.outerFrame, relief=tk.FLAT, pady=-2)
        self.updateStateChangeButton()

        # Lay out widgets
        self.stateChangeButton.grid(row=0, column=0, sticky=tk.NW)
        self.grid(row=1, column=0, sticky=tk.NSEW, collapsableInner=True)

        if collapsed:
            self.collapse()
        else:
            self.expand()

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
        # Return whether frame is in collapsed or expanded state
        return self._isCollapsed

    def collapse(self):
        # Set boolean isCollapsed flag to False
        self._isCollapsed = True
        # Update state change button
        self.updateStateChangeButton()
        # Hide frame
        self.grid_forget(collapsableInner=True)
        # If present, run user-supplied collapse callback
        if self.collapseFunction is not None:
            self.collapseFunction(self)

    def expand(self):
        # Set boolean isCollapsed flag to False
        self._isCollapsed = False
        # Update state change button
        self.updateStateChangeButton()
        # Show frame
        self.grid(collapsableInner=True)
        # If present, run user-supplied expand callback
        if self.expandFunction is not None:
            self.expandFunction(self)

    def after(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().after(*args, **kwargs)
        else:
            return self.outerFrame.after(*args, **kwargs)
    def after_cancel(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().after_cancel(*args, **kwargs)
        else:
            return self.outerFrame.after_cancel(*args, **kwargs)
    def after_idle(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().after_idle(*args, **kwargs)
        else:
            return self.outerFrame.after_idle(*args, **kwargs)
    def anchor(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().anchor(*args, **kwargs)
        else:
            return self.outerFrame.anchor(*args, **kwargs)
    def bbox(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().bbox(*args, **kwargs)
        else:
            return self.outerFrame.bbox(*args, **kwargs)
    def bell(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().bell(*args, **kwargs)
        else:
            return self.outerFrame.bell(*args, **kwargs)
    def bind(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().bind(*args, **kwargs)
        else:
            return self.outerFrame.bind(*args, **kwargs)
    def bind_all(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().bind_all(*args, **kwargs)
        else:
            return self.outerFrame.bind_all(*args, **kwargs)
    def bind_class(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().bind_class(*args, **kwargs)
        else:
            return self.outerFrame.bind_class(*args, **kwargs)
    def bindtags(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().bindtags(*args, **kwargs)
        else:
            return self.outerFrame.bindtags(*args, **kwargs)
    def cget(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().cget(*args, **kwargs)
        else:
            return self.outerFrame.cget(*args, **kwargs)
    def children(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().children(*args, **kwargs)
        else:
            return self.outerFrame.children(*args, **kwargs)
    def clipboard_append(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().clipboard_append(*args, **kwargs)
        else:
            return self.outerFrame.clipboard_append(*args, **kwargs)
    def clipboard_clear(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().clipboard_clear(*args, **kwargs)
        else:
            return self.outerFrame.clipboard_clear(*args, **kwargs)
    def clipboard_get(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().clipboard_get(*args, **kwargs)
        else:
            return self.outerFrame.clipboard_get(*args, **kwargs)
    def columnconfigure(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().columnconfigure(*args, **kwargs)
        else:
            return self.outerFrame.columnconfigure(*args, **kwargs)
    def config(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().config(*args, **kwargs)
        else:
            return self.outerFrame.config(*args, **kwargs)
    def configure(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().configure(*args, **kwargs)
        else:
            return self.outerFrame.configure(*args, **kwargs)
    def deletecommand(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().deletecommand(*args, **kwargs)
        else:
            return self.outerFrame.deletecommand(*args, **kwargs)
    # def destroy(self, *args, collapsableInner=False, **kwargs):
    #if collapsableInner:
    #     return super().     destroy(*args, **kwargs)
    # else:
    #     return self.outerFrame.destroy(*args, **kwargs)
    def event_add(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().event_add(*args, **kwargs)
        else:
            return self.outerFrame.event_add(*args, **kwargs)
    def event_delete(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().event_delete(*args, **kwargs)
        else:
            return self.outerFrame.event_delete(*args, **kwargs)
    def event_generate(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().event_generate(*args, **kwargs)
        else:
            return self.outerFrame.event_generate(*args, **kwargs)
    def event_info(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().event_info(*args, **kwargs)
        else:
            return self.outerFrame.event_info(*args, **kwargs)
    def focus(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().focus(*args, **kwargs)
        else:
            return self.outerFrame.focus(*args, **kwargs)
    def focus_displayof(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().focus_displayof(*args, **kwargs)
        else:
            return self.outerFrame.focus_displayof(*args, **kwargs)
    def focus_force(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().focus_force(*args, **kwargs)
        else:
            return self.outerFrame.focus_force(*args, **kwargs)
    def focus_get(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().focus_get(*args, **kwargs)
        else:
            return self.outerFrame.focus_get(*args, **kwargs)
    def focus_lastfor(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().focus_lastfor(*args, **kwargs)
        else:
            return self.outerFrame.focus_lastfor(*args, **kwargs)
    def focus_set(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().focus_set(*args, **kwargs)
        else:
            return self.outerFrame.focus_set(*args, **kwargs)
    def forget(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().forget(*args, **kwargs)
        else:
            return self.outerFrame.forget(*args, **kwargs)
    def getboolean(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().getboolean(*args, **kwargs)
        else:
            return self.outerFrame.getboolean(*args, **kwargs)
    def getdouble(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().getdouble(*args, **kwargs)
        else:
            return self.outerFrame.getdouble(*args, **kwargs)
    def getint(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().getint(*args, **kwargs)
        else:
            return self.outerFrame.getint(*args, **kwargs)
    def getvar(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().getvar(*args, **kwargs)
        else:
            return self.outerFrame.getvar(*args, **kwargs)
    def grab_current(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grab_current(*args, **kwargs)
        else:
            return self.outerFrame.grab_current(*args, **kwargs)
    def grab_release(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grab_release(*args, **kwargs)
        else:
            return self.outerFrame.grab_release(*args, **kwargs)
    def grab_set(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grab_set(*args, **kwargs)
        else:
            return self.outerFrame.grab_set(*args, **kwargs)
    def grab_set_global(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grab_set_global(*args, **kwargs)
        else:
            return self.outerFrame.grab_set_global(*args, **kwargs)
    def grab_status(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grab_status(*args, **kwargs)
        else:
            return self.outerFrame.grab_status(*args, **kwargs)
    def grid(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grid(*args, **kwargs)
        else:
            return self.outerFrame.grid(*args, **kwargs)
    def grid_anchor(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grid_anchor(*args, **kwargs)
        else:
            return self.outerFrame.grid_anchor(*args, **kwargs)
    def grid_bbox(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grid_bbox(*args, **kwargs)
        else:
            return self.outerFrame.grid_bbox(*args, **kwargs)
    def grid_columnconfigure(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grid_columnconfigure(*args, **kwargs)
        else:
            return self.outerFrame.grid_columnconfigure(*args, **kwargs)
    def grid_configure(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grid_configure(*args, **kwargs)
        else:
            return self.outerFrame.grid_configure(*args, **kwargs)
    def grid_forget(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grid_forget(*args, **kwargs)
        else:
            return self.outerFrame.grid_forget(*args, **kwargs)
    def grid_info(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grid_info(*args, **kwargs)
        else:
            return self.outerFrame.grid_info(*args, **kwargs)
    def grid_location(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grid_location(*args, **kwargs)
        else:
            return self.outerFrame.grid_location(*args, **kwargs)
    def grid_propagate(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grid_propagate(*args, **kwargs)
        else:
            return self.outerFrame.grid_propagate(*args, **kwargs)
    def grid_remove(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grid_remove(*args, **kwargs)
        else:
            return self.outerFrame.grid_remove(*args, **kwargs)
    def grid_rowconfigure(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grid_rowconfigure(*args, **kwargs)
        else:
            return self.outerFrame.grid_rowconfigure(*args, **kwargs)
    def grid_size(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grid_size(*args, **kwargs)
        else:
            return self.outerFrame.grid_size(*args, **kwargs)
    def grid_slaves(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().grid_slaves(*args, **kwargs)
        else:
            return self.outerFrame.grid_slaves(*args, **kwargs)
    def image_names(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().image_names(*args, **kwargs)
        else:
            return self.outerFrame.image_names(*args, **kwargs)
    def image_types(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().image_types(*args, **kwargs)
        else:
            return self.outerFrame.image_types(*args, **kwargs)
    def info(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().info(*args, **kwargs)
        else:
            return self.outerFrame.info(*args, **kwargs)
    def keys(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().keys(*args, **kwargs)
        else:
            return self.outerFrame.keys(*args, **kwargs)
    def lift(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().lift(*args, **kwargs)
        else:
            return self.outerFrame.lift(*args, **kwargs)
    def location(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().location(*args, **kwargs)
        else:
            return self.outerFrame.location(*args, **kwargs)
    def lower(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().lower(*args, **kwargs)
        else:
            return self.outerFrame.lower(*args, **kwargs)
    def mainloop(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().mainloop(*args, **kwargs)
        else:
            return self.outerFrame.mainloop(*args, **kwargs)
    def master(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().master(*args, **kwargs)
        else:
            return self.outerFrame.master(*args, **kwargs)
    def nametowidget(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().nametowidget(*args, **kwargs)
        else:
            return self.outerFrame.nametowidget(*args, **kwargs)
    def option_add(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().option_add(*args, **kwargs)
        else:
            return self.outerFrame.option_add(*args, **kwargs)
    def option_clear(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().option_clear(*args, **kwargs)
        else:
            return self.outerFrame.option_clear(*args, **kwargs)
    def option_get(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().option_get(*args, **kwargs)
        else:
            return self.outerFrame.option_get(*args, **kwargs)
    def option_readfile(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().option_readfile(*args, **kwargs)
        else:
            return self.outerFrame.option_readfile(*args, **kwargs)
    def pack(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().pack(*args, **kwargs)
        else:
            return self.outerFrame.pack(*args, **kwargs)
    def pack_configure(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().pack_configure(*args, **kwargs)
        else:
            return self.outerFrame.pack_configure(*args, **kwargs)
    def pack_forget(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().pack_forget(*args, **kwargs)
        else:
            return self.outerFrame.pack_forget(*args, **kwargs)
    def pack_info(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().pack_info(*args, **kwargs)
        else:
            return self.outerFrame.pack_info(*args, **kwargs)
    def pack_propagate(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().pack_propagate(*args, **kwargs)
        else:
            return self.outerFrame.pack_propagate(*args, **kwargs)
    def pack_slaves(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().pack_slaves(*args, **kwargs)
        else:
            return self.outerFrame.pack_slaves(*args, **kwargs)
    def place(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().place(*args, **kwargs)
        else:
            return self.outerFrame.place(*args, **kwargs)
    def place_configure(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().place_configure(*args, **kwargs)
        else:
            return self.outerFrame.place_configure(*args, **kwargs)
    def place_forget(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().place_forget(*args, **kwargs)
        else:
            return self.outerFrame.place_forget(*args, **kwargs)
    def place_info(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().place_info(*args, **kwargs)
        else:
            return self.outerFrame.place_info(*args, **kwargs)
    def place_slaves(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().place_slaves(*args, **kwargs)
        else:
            return self.outerFrame.place_slaves(*args, **kwargs)
    def propagate(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().propagate(*args, **kwargs)
        else:
            return self.outerFrame.propagate(*args, **kwargs)
    def quit(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().quit(*args, **kwargs)
        else:
            return self.outerFrame.quit(*args, **kwargs)
    def register(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().register(*args, **kwargs)
        else:
            return self.outerFrame.register(*args, **kwargs)
    def rowconfigure(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().rowconfigure(*args, **kwargs)
        else:
            return self.outerFrame.rowconfigure(*args, **kwargs)
    def selection_clear(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().selection_clear(*args, **kwargs)
        else:
            return self.outerFrame.selection_clear(*args, **kwargs)
    def selection_get(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().selection_get(*args, **kwargs)
        else:
            return self.outerFrame.selection_get(*args, **kwargs)
    def selection_handle(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().selection_handle(*args, **kwargs)
        else:
            return self.outerFrame.selection_handle(*args, **kwargs)
    def selection_own(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().selection_own(*args, **kwargs)
        else:
            return self.outerFrame.selection_own(*args, **kwargs)
    def selection_own_get(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().selection_own_get(*args, **kwargs)
        else:
            return self.outerFrame.selection_own_get(*args, **kwargs)
    def send(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().send(*args, **kwargs)
        else:
            return self.outerFrame.send(*args, **kwargs)
    def setvar(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().setvar(*args, **kwargs)
        else:
            return self.outerFrame.setvar(*args, **kwargs)
    def size(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().size(*args, **kwargs)
        else:
            return self.outerFrame.size(*args, **kwargs)
    def slaves(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().slaves(*args, **kwargs)
        else:
            return self.outerFrame.slaves(*args, **kwargs)
    def tk(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().tk(*args, **kwargs)
        else:
            return self.outerFrame.tk(*args, **kwargs)
    def tk_bisque(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().tk_bisque(*args, **kwargs)
        else:
            return self.outerFrame.tk_bisque(*args, **kwargs)
    def tk_focusFollowsMouse(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().tk_focusFollowsMouse(*args, **kwargs)
        else:
            return self.outerFrame.tk_focusFollowsMouse(*args, **kwargs)
    def tk_focusNext(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().tk_focusNext(*args, **kwargs)
        else:
            return self.outerFrame.tk_focusNext(*args, **kwargs)
    def tk_focusPrev(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().tk_focusPrev(*args, **kwargs)
        else:
            return self.outerFrame.tk_focusPrev(*args, **kwargs)
    def tk_setPalette(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().tk_setPalette(*args, **kwargs)
        else:
            return self.outerFrame.tk_setPalette(*args, **kwargs)
    def tk_strictMotif(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().tk_strictMotif(*args, **kwargs)
        else:
            return self.outerFrame.tk_strictMotif(*args, **kwargs)
    def tkraise(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().tkraise(*args, **kwargs)
        else:
            return self.outerFrame.tkraise(*args, **kwargs)
    def unbind(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().unbind(*args, **kwargs)
        else:
            return self.outerFrame.unbind(*args, **kwargs)
    def unbind_all(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().unbind_all(*args, **kwargs)
        else:
            return self.outerFrame.unbind_all(*args, **kwargs)
    def unbind_class(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().unbind_class(*args, **kwargs)
        else:
            return self.outerFrame.unbind_class(*args, **kwargs)
    def update(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().update(*args, **kwargs)
        else:
            return self.outerFrame.update(*args, **kwargs)
    def update_idletasks(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().update_idletasks(*args, **kwargs)
        else:
            return self.outerFrame.update_idletasks(*args, **kwargs)
    def wait_variable(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().wait_variable(*args, **kwargs)
        else:
            return self.outerFrame.wait_variable(*args, **kwargs)
    def wait_visibility(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().wait_visibility(*args, **kwargs)
        else:
            return self.outerFrame.wait_visibility(*args, **kwargs)
    def wait_window(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().wait_window(*args, **kwargs)
        else:
            return self.outerFrame.wait_window(*args, **kwargs)
    def waitvar(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().waitvar(*args, **kwargs)
        else:
            return self.outerFrame.waitvar(*args, **kwargs)
    def widgetName(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().widgetName(*args, **kwargs)
        else:
            return self.outerFrame.widgetName(*args, **kwargs)
    def winfo_atom(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_atom(*args, **kwargs)
        else:
            return self.outerFrame.winfo_atom(*args, **kwargs)
    def winfo_atomname(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_atomname(*args, **kwargs)
        else:
            return self.outerFrame.winfo_atomname(*args, **kwargs)
    def winfo_cells(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_cells(*args, **kwargs)
        else:
            return self.outerFrame.winfo_cells(*args, **kwargs)
    def winfo_children(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_children(*args, **kwargs)
        else:
            return self.outerFrame.winfo_children(*args, **kwargs)
    def winfo_class(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_class(*args, **kwargs)
        else:
            return self.outerFrame.winfo_class(*args, **kwargs)
    def winfo_colormapfull(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_colormapfull(*args, **kwargs)
        else:
            return self.outerFrame.winfo_colormapfull(*args, **kwargs)
    def winfo_containing(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_containing(*args, **kwargs)
        else:
            return self.outerFrame.winfo_containing(*args, **kwargs)
    def winfo_depth(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_depth(*args, **kwargs)
        else:
            return self.outerFrame.winfo_depth(*args, **kwargs)
    def winfo_exists(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_exists(*args, **kwargs)
        else:
            return self.outerFrame.winfo_exists(*args, **kwargs)
    def winfo_fpixels(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_fpixels(*args, **kwargs)
        else:
            return self.outerFrame.winfo_fpixels(*args, **kwargs)
    def winfo_geometry(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_geometry(*args, **kwargs)
        else:
            return self.outerFrame.winfo_geometry(*args, **kwargs)
    def winfo_height(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_height(*args, **kwargs)
        else:
            return self.outerFrame.winfo_height(*args, **kwargs)
    def winfo_id(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_id(*args, **kwargs)
        else:
            return self.outerFrame.winfo_id(*args, **kwargs)
    def winfo_interps(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_interps(*args, **kwargs)
        else:
            return self.outerFrame.winfo_interps(*args, **kwargs)
    def winfo_ismapped(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_ismapped(*args, **kwargs)
        else:
            return self.outerFrame.winfo_ismapped(*args, **kwargs)
    def winfo_manager(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_manager(*args, **kwargs)
        else:
            return self.outerFrame.winfo_manager(*args, **kwargs)
    def winfo_name(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_name(*args, **kwargs)
        else:
            return self.outerFrame.winfo_name(*args, **kwargs)
    def winfo_parent(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_parent(*args, **kwargs)
        else:
            return self.outerFrame.winfo_parent(*args, **kwargs)
    def winfo_pathname(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_pathname(*args, **kwargs)
        else:
            return self.outerFrame.winfo_pathname(*args, **kwargs)
    def winfo_pixels(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_pixels(*args, **kwargs)
        else:
            return self.outerFrame.winfo_pixels(*args, **kwargs)
    def winfo_pointerx(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_pointerx(*args, **kwargs)
        else:
            return self.outerFrame.winfo_pointerx(*args, **kwargs)
    def winfo_pointerxy(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_pointerxy(*args, **kwargs)
        else:
            return self.outerFrame.winfo_pointerxy(*args, **kwargs)
    def winfo_pointery(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_pointery(*args, **kwargs)
        else:
            return self.outerFrame.winfo_pointery(*args, **kwargs)
    def winfo_reqheight(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_reqheight(*args, **kwargs)
        else:
            return self.outerFrame.winfo_reqheight(*args, **kwargs)
    def winfo_reqwidth(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_reqwidth(*args, **kwargs)
        else:
            return self.outerFrame.winfo_reqwidth(*args, **kwargs)
    def winfo_rgb(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_rgb(*args, **kwargs)
        else:
            return self.outerFrame.winfo_rgb(*args, **kwargs)
    def winfo_rootx(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_rootx(*args, **kwargs)
        else:
            return self.outerFrame.winfo_rootx(*args, **kwargs)
    def winfo_rooty(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_rooty(*args, **kwargs)
        else:
            return self.outerFrame.winfo_rooty(*args, **kwargs)
    def winfo_screen(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_screen(*args, **kwargs)
        else:
            return self.outerFrame.winfo_screen(*args, **kwargs)
    def winfo_screencells(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_screencells(*args, **kwargs)
        else:
            return self.outerFrame.winfo_screencells(*args, **kwargs)
    def winfo_screendepth(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_screendepth(*args, **kwargs)
        else:
            return self.outerFrame.winfo_screendepth(*args, **kwargs)
    def winfo_screenheight(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_screenheight(*args, **kwargs)
        else:
            return self.outerFrame.winfo_screenheight(*args, **kwargs)
    def winfo_screenmmheight(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_screenmmheight(*args, **kwargs)
        else:
            return self.outerFrame.winfo_screenmmheight(*args, **kwargs)
    def winfo_screenmmwidth(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_screenmmwidth(*args, **kwargs)
        else:
            return self.outerFrame.winfo_screenmmwidth(*args, **kwargs)
    def winfo_screenvisual(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_screenvisual(*args, **kwargs)
        else:
            return self.outerFrame.winfo_screenvisual(*args, **kwargs)
    def winfo_screenwidth(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_screenwidth(*args, **kwargs)
        else:
            return self.outerFrame.winfo_screenwidth(*args, **kwargs)
    def winfo_server(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_server(*args, **kwargs)
        else:
            return self.outerFrame.winfo_server(*args, **kwargs)
    def winfo_toplevel(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_toplevel(*args, **kwargs)
        else:
            return self.outerFrame.winfo_toplevel(*args, **kwargs)
    def winfo_viewable(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_viewable(*args, **kwargs)
        else:
            return self.outerFrame.winfo_viewable(*args, **kwargs)
    def winfo_visual(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_visual(*args, **kwargs)
        else:
            return self.outerFrame.winfo_visual(*args, **kwargs)
    def winfo_visualid(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_visualid(*args, **kwargs)
        else:
            return self.outerFrame.winfo_visualid(*args, **kwargs)
    def winfo_visualsavailable(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_visualsavailable(*args, **kwargs)
        else:
            return self.outerFrame.winfo_visualsavailable(*args, **kwargs)
    def winfo_vrootheight(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_vrootheight(*args, **kwargs)
        else:
            return self.outerFrame.winfo_vrootheight(*args, **kwargs)
    def winfo_vrootwidth(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_vrootwidth(*args, **kwargs)
        else:
            return self.outerFrame.winfo_vrootwidth(*args, **kwargs)
    def winfo_vrootx(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_vrootx(*args, **kwargs)
        else:
            return self.outerFrame.winfo_vrootx(*args, **kwargs)
    def winfo_vrooty(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_vrooty(*args, **kwargs)
        else:
            return self.outerFrame.winfo_vrooty(*args, **kwargs)
    def winfo_width(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_width(*args, **kwargs)
        else:
            return self.outerFrame.winfo_width(*args, **kwargs)
    def winfo_x(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_x(*args, **kwargs)
        else:
            return self.outerFrame.winfo_x(*args, **kwargs)
    def winfo_y(self, *args, collapsableInner=False, **kwargs):
        if collapsableInner:
            return super().winfo_y(*args, **kwargs)
        else:
            return self.outerFrame.winfo_y(*args, **kwargs)

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
    cf1 = CollapsableFrame(mainFrame, collapseText='Collapse the frame!',
        expandText='Expand the frame!',borderwidth=3, relief=tk.SUNKEN)
    # Display the collapsable frame
    cf1.grid(row=1, column=0, sticky=tk.NSEW)
    cf1.grid_columnconfigure(0, weight=1)
    # A label that's in the collapsable frame.
    innerLabel1_1 = tk.Label(cf1, text='Hi there, I am inside a collapsable frame!')
    innerLabel2_1 = tk.Label(cf1, text='Me too!')
    # Display inner label
    innerLabel1_1.grid(row=1, column=0)
    innerLabel2_1.grid(row=2, column=0)

    # Create another collapsable frame
    cf2 = CollapsableFrame(mainFrame, collapseText='Another collapsable frame',
        borderwidth=3, relief=tk.SUNKEN)
    # Display the collapsable frame
    cf2.grid(row=2, column=0, sticky=tk.NSEW)
    cf2.grid_columnconfigure(0, weight=1)
    # A label that's in the collapsable frame.
    innerLabel1_2 = tk.Label(cf2, text='Look I am a thing in a collapsable frame')
    innerLabel2_2 = tk.Label(cf2, text='Wheeee!')
    # Display inner label
    innerLabel1_2.grid(row=1, column=0)
    innerLabel2_2.grid(row=2, column=0)

    # Create & display a label that is outside the collapsable frame
    outerLabel = tk.Label(mainFrame, text='That there is a collapsable frame! \/\/\/')
    outerLabel.grid(row=0, column=0)

    # Start main GUI loop
    root.mainloop()
