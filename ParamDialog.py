import tkinter as tk
import tkinter.ttk as ttk
import math

class Param():
    TEXT='text'
    MONOCHOICE='monochoice'
    MULTICHOICE='multichoice'

    def __init__(self, name='unnamedParam', widgetType=TEXT, options=[], default=None, parser=lambda x:x, description=None):
        # parent = a tkinter container that widgets should belong to
        # name = the name of the parameter
        # widgetType = one of Param.TEXT, Param.MONOCHOICE, Param.MULTICHOICE
        # options = a list of options to select from (not relevant for TEXT types)
        # default = default value. If default is supplied for MONOCHOICE,
        #           default must be one of the options. or none. If default is
        #           supplied for MULTICHOICE types, default either be None or
        #           a list of one or more of the supplied options
        # parser = function to parse input string, for example 'float'
        if type in [Param.MONOCHOICE, Param.MULTICHOICE]:
            if default is not None:
                if default not in options:
                    raise IndexError('Supplied default must be one of the supplied options')
        self.name = name
        self.widgetType = widgetType
        self.options = options
        self.description = description
        self.default = default
        self.parser = parser
        self.var = None
        self.widgets = []
        self.mainFrame = None
        self.widgetFrame = None
        self.label = None

    def createWidgets(self, parent, maxHeight=None):
        self.mainFrame = ttk.LabelFrame(parent, text=self.name)
        self.widgetFrame = ttk.Frame(self.mainFrame)
        if self.description is not None and len(self.description) > 0:
            labelWidth = max([len(self.name), 7]) * 10
            self.label = ttk.Label(self.mainFrame, text=self.description, wraplength=labelWidth)
        if self.widgetType == Param.TEXT:
            self.var = tk.StringVar()
            entry = ttk.Entry(self.widgetFrame, textvariable=self.var)
            entry.grid(row=0, column=0, sticky=tk.NW)
            self.widgets.append(entry)
            if self.default is None:
                self.var.set('')
            else:
                self.var.set(self.default)
        elif self.widgetType == Param.MONOCHOICE:
            self.var = tk.StringVar()
            for k, option in enumerate(self.options):
                rbutton = ttk.Radiobutton(self.widgetFrame, text=option, variable=self.var, value=option)
                if maxHeight is None:
                    row = k
                    column = 0
                else:
                    row = (k % maxHeight)
                    column = k // maxHeight
                rbutton.grid(row=row, column=column, sticky=tk.NW)
                self.widgets.append(rbutton)
            if self.default is not None:
                self.var.set(self.default)
            else:
                self.var.set(self.options[0])
        elif self.widgetType == Param.MULTICHOICE:
            self.var = []
            for k, option in enumerate(self.options):
                var = tk.StringVar()
                self.var.append(var)
                cbutton = ttk.Checkbutton(self.widgetFrame, text=option, variable=var, onvalue=option, offvalue='')
                if (self.default is not None) and (option in self.default):
                    self.var[k].set(option)
                else:
                    self.var[k].set('')
                if maxHeight is None:
                    row = k
                    column = 0
                else:
                    row = ((k+1) % maxHeight)
                    column = (k+1) // maxHeight
                cbutton.grid(row=row, column=column, sticky=tk.NW)
                self.widgets.append(cbutton)

            self.selectAllVar = tk.IntVar()
            self.selectAllButton = ttk.Checkbutton(self.widgetFrame, text="Select all", variable=self.selectAllVar, onvalue=1, offvalue=0)
            self.selectAllButton.grid(row=0, column=0, sticky=tk.NW)
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
        if self.label is not None:
            self.label.grid(row=0, column=0, sticky=tk.NW)
        self.widgetFrame.grid(row=1, column=0, sticky=tk.NW)


    def grid(self, *args, **kwargs):
        if self.mainFrame is not None:
            self.mainFrame.grid(*args, **kwargs)
        else:
            raise AttributeError('You must call createWidgets on this Param object before calling grid')

    def get(self):
        if self.mainFrame is not None:
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

    def __init__(self, parent, params=[], title=None, arrangement=HORIZONTAL, maxHeight=None):
        # params should be a list of Param objects
        # maxHeight is the maximum number of parameter options that can be stacked before wrapping horizontally. Leave as "None" to disable wrapping.
        tk.Toplevel.__init__(self, parent)
        self.transient(parent)
        if title:
            self.title(title)

        self.parent = parent
        self.arrangement = arrangement
        self.maxHeight = maxHeight
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
        if self.arrangement == ParamDialog.BOX:
            nW, nH = getOptimalBoxGrid(len(self.params))
        for k, param in enumerate(self.params):
            param.createWidgets(self.paramFrame, maxHeight=self.maxHeight)
            if self.arrangement == ParamDialog.HORIZONTAL:
                row=0; col=k;
                self.paramFrame.columnconfigure(col, weight=1)
            elif self.arrangement == ParamDialog.VERTICAL:
                row=k; col=0;
                self.paramFrame.rowconfigure(row, weight=1)
            elif self.arrangement == ParamDialog.BOX:
                row = k // nW
                col = k % nW
                self.paramFrame.rowconfigure(row, weight=1)
                self.paramFrame.columnconfigure(col, weight=1)
                # raise NotImplementedError("Box arrangement has not been implemented yet.")
            else:
                raise NameError("Unknown arrangement type: "+str(self.arrangement))
            param.grid(row=row, column=col, sticky=tk.NSEW)

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
            # print("Got", param.name, "=", self.results[param.name], type(self.results[param.name]))

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

def getOptimalBoxGrid(N, maxHeight=None):
    if N == 0:
        return (0,0)
    h = round(math.sqrt(N))
    if maxHeight is not None:
        h = min(maxHeight, h)
    w = math.ceil(N/h)
    return (w, h)
