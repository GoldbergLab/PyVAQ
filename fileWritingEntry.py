# A tkinter widget for selecting a directory and base filename for writing files
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askdirectory
import os
import re
import unicodedata

class FileWritingEntry(ttk.LabelFrame):
    def __init__(self, master, *args, defaultDirectory='', defaultBaseFileName='', purposeText="", **kwargs):
        ttk.LabelFrame.__init__(self, master, *args, **kwargs)
        self.master = master

        self.purposeText = purposeText

        self.baseFileNameFrame =    ttk.LabelFrame(self, text="Base Filename", style='SingleContainer.TLabelframe')
        self.baseFileNameVar =      tk.StringVar(self.master); self.baseFileNameVar.set(defaultBaseFileName)
        self.baseFileNameEntry =    ttk.Entry(self.baseFileNameFrame, width=12, textvariable=self.baseFileNameVar)

        self.directoryFrame =       ttk.LabelFrame(self, text="Directory", style='SingleContainer.TLabelframe')
        self.directoryVar =         tk.StringVar(self.master); self.directoryVar.set(defaultDirectory)
        self.directoryEntry =       ttk.Entry(self.directoryFrame, width=18, textvariable=self.directoryVar, style='ValidDirectory.TEntry')
        self.directoryButton =      ttk.Button(self.directoryFrame, text='O', width=2, command=self.selectWriteDirectory)

        self.directoryChangeHandler = lambda:None
        self.baseFileNameChangeHandler = lambda:None

        self.directoryEntry.bind('<FocusOut>', self.directoryChangeMetaHandler)
        self.baseFileNameEntry.bind('<FocusOut>', self.baseFileNameChangeMetaHandler)

        self.directoryFrame.grid(row=0, column=0, sticky=tk.NSEW)
        self.directoryEntry.grid(row=0, column=0)
        self.directoryButton.grid(row=0, column=1)

        self.baseFileNameFrame.grid(row=0, column=1, sticky=tk.NSEW)
        self.baseFileNameEntry.grid()

        self.checkForValidDirectory()

    def getDirectory(self):
        return self.directoryVar.get()

    def getBaseFileName(self):
        return self.baseFileNameVar.get()

    def directoryChangeMetaHandler(self, *args):
        self.checkForValidDirectory()
        self.directoryChangeHandler()

    def checkForValidDirectory(self):
        newDir = self.directoryVar.get()
        if len(newDir) == 0 or os.path.isdir(newDir):
            self.directoryEntry['style'] = 'ValidDirectory.TEntry'
            print("Valid directory")
        else:
            self.directoryEntry['style'] = 'InvalidDirectory.TEntry'
            print("Invalid directory")

    def baseFileNameChangeMetaHandler(self, *args):
        # Regularize filename
        self.baseFileNameVar.set(slugify(self.baseFileNameVar.get()))
        self.baseFileNameChangeHandler()

    def setDirectoryChangeHandler(self, function):
        self.directoryChangeHandler = function

    def setBaseFileNameChangeHandler(self, function):
        self.baseFileNameChangeHandler = function

    def selectWriteDirectory(self, *args):
        if len(self.purposeText) > 0:
            title = "Choose a directory for {purpose}.".format(purpose=self.purposeText)
        else:
            title = "Choose a directory"

        directory = askdirectory(
#            initialdir = ,
            mustexist = False,
            title = title
        )
        if len(directory) > 0:
            self.directoryVar.set(directory)
            self.directoryEntry.xview_moveto(0.5)
            self.directoryEntry.update_idletasks()
            self.directoryChangeMetaHandler()

def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.

    Adapter from Django utils
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)

if __name__ == "__main__":
    WIDGET_COLORS = [
    '#050505', # near black
    '#e6f5ff', # very light blue
    '#c1ffc1', # light green
    '#FFC1C1'  # light red
    ]
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('default')
    style.configure('ValidDirectory.TEntry', fieldbackground=WIDGET_COLORS[2])
    style.configure('InvalidDirectory.TEntry', fieldbackground=WIDGET_COLORS[3])
    f = FileWritingEntry(root)
    f.grid()
    root.mainloop()
