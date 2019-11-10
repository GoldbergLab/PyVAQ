import tkinter as tk
import tkinter.ttk as ttk

class Param():
    TEXT='text'
    MONOCHOICE='monochoice'
    MULTICHOICE='multichoice'

    def __init__(self, name='unnamedParam', widgetType=TEXT, options=[], default='default', parser=lambda x:x):
        # parent = a tkinter container that widgets should belong to
        # name = the name of the parameter
        # widgetType = one of Param.TEXT, Param.MONOCHOICE, Param.MULTICHOICE
        # options = a list of options to select from (not relevant for TEXT types)
        # default = default value. If options are supplied (for MONOCHOICE and MULTICHOICE types, default either be None or one of the supplied options)
        # parser = function to parse input string, for example 'float'
        if type in [Param.MONOCHOICE, Param.MULTICHOICE]:
            if default is not None:
                if default not in options:
                    raise IndexError('Supplied default must be one of the supplied options')
        self.name = name
        self.widgetType = widgetType
        self.options = options
        self.default = default
        self.parser = parser
        self.var = None
        self.widgets = []
        self.frame = None

    def createWidgets(self, parent):
        self.frame = ttk.LabelFrame(parent, text=self.name)
        if self.widgetType == Param.TEXT:
            self.var = tk.StringVar()
            entry = ttk.Entry(self.frame)
            entry.grid()
            self.widgets.append(entry)
            self.var.set(self.default)
        elif self.widgetType == Param.MONOCHOICE:
            self.var = tk.StringVar()
            for k, option in enumerate(self.options):
                rbutton = ttk.Radiobutton(self.frame, text=option, variable=self.var, value=option)
                rbutton.grid(row=k, column=0)
                self.widgets.append(rbutton)
                self.var.set(self.default)
        elif self.widgetType == Param.MULTICHOICE:
            self.var = []
            for k, option in enumerate(self.options):
                var = tk.StringVar()
                self.var.append(var)
                cbutton = ttk.Checkbutton(self.frame, text=option, variable=var, onvalue=option, offvalue='')
                cbutton.grid(row=k+1, column=0)
                self.widgets.append(cbutton)

            self.selectAllVar = tk.IntVar()
            self.selectAllButton = ttk.Checkbutton(self.frame, text="Select all", variable=self.selectAllVar, onvalue=1, offvalue=0)
            self.selectAllButton.grid(row=0, column=0)
            self.selectAllVar.set(0)
            def selectAllOrNoneCallbackFactory(savar, cbuttons, vars):
                def callback(*args):
                    if savar.get():
                        for cbutton, var in zip(cbuttons, vars):
                            var.set(cbutton.cget('onvalue'))
                    else:
                        for cbutton, var in zip(cbuttons, vars):
                            var.set(cbutton.cget('offvalue'))
                return callback
            saCallback = selectAllOrNoneCallbackFactory(self.selectAllVar, self.widgets, self.var)
            self.selectAllVar.trace('w', saCallback)

    def grid(self, *args, **kwargs):
        if self.frame is not None:
            self.frame.grid(*args, **kwargs)
        else:
            raise AttributeError('You must call createWidgets on this Param object before calling grid')

    def get(self):
        if self.frame is not None:
            if self.widgetType in [Param.TEXT, Param.MONOCHOICE]:
                return self.parser(self.var.get())
            elif self.widgetType in [Param.MULTICHOICE]:
                return [self.parser(var.get()) for var in self.var if len(var.get()) > 0]
        else:
            raise AttributeError('You must call createWidgets on this Param object before calling get')

class ParamDialog(tk.Toplevel):
    HORIZONTAL='h'
    VERTICAL='v'
    BOX='b'

    def __init__(self, parent, params=[], title=None, arrangement=HORIZONTAL):
        # params should be a list of Param objects
        tk.Toplevel.__init__(self, parent)
        self.transient(parent)
        if title:
            self.title(title)

        self.parent = parent
        self.arrangement = arrangement
        self.params = params
        self.results = None

        self.mainFrame = ttk.Frame(self)
        self.paramFrame = ttk.Frame(self.mainFrame)
        self.buttonFrame = ttk.Frame(self.mainFrame)
        self.parameterWidgets = {}
        self.createParameterWidgets()

        self.grab_set()

        self.mainFrame.grid()
        self.paramFrame.grid(row=0, sticky=tk.NSEW)
        self.buttonFrame.grid(row=1, sticky=tk.NSEW)

        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
                                  parent.winfo_rooty()+50))
        self.focus_set()
        self.wait_window(self)

    def createParameterWidgets(self):
        # Create widgets for inputting parameters
        for k, param in enumerate(self.params):
            param.createWidgets(self.paramFrame)
            if self.arrangement == ParamDialog.HORIZONTAL:
                self.paramFrame.columnconfigure(k, weight=1)
                row=0; column=k;
            elif self.arrangement == ParamDialog.VERTICAL:
                self.paramFrame.rowconfigure(k, weight=1)
                row=k; column=0;
            elif self.arrangement == ParamDialog.BOX:
                raise NotImplementedError("Box arrangement has not been implemented yet.")
            else:
                raise NameError("Unknown arrangement type: "+str(self.arrangement))
            param.grid(row=row, column=column, sticky=tk.NSEW)

        okButton = ttk.Button(self.buttonFrame, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        okButton.grid(row=0, column=0)
        cancelButton = ttk.Button(self.buttonFrame, text="Cancel", width=10, command=self.cancel)
        cancelButton.grid(row=0, column=1)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def collectParams(self):
        self.results = {}
        for param in self.params:
            self.results[param.name] = param.get()

    def ok(self, event=None):
        # if not self.validate():
        #     self.initial_focus.focus_set() # put focus back
        #     return
        self.collectParams()
        self.withdraw()
        self.update_idletasks()
        # self.apply()
        self.parent.focus_set()
        self.destroy()

    def cancel(self, event=None):
        # put focus back to the parent window
        self.results = None
        self.parent.focus_set()
        self.destroy()

    # def validate(self):
    #     return 1 # override
    #
    # def apply(self):
    #     pass # override
