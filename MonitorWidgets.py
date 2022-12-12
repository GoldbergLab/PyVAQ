import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
from PIL import Image, ImageTk
#from SharedImageQueue import SharedImageSender
from scipy.signal import butter, lfilter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
# from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from fileWritingEntry import FileWritingEntry
import cv2
import CameraUtilities as cu

WIDGET_COLORS = [
    '#050505', # near black
    '#e6f5ff', # very light blue
    '#c1ffc1', # light green
    '#FFC1C1'  # light red
]
LINE_STYLES = [c+'-' for c in 'bykcmgr']

with Image.open(r'Resources\NoImages_000.png') as NO_IMAGES_IMAGE:
    NO_IMAGES_IMAGE.load()

class AudioMonitor(ttk.LabelFrame):
    def __init__(self, *args, historyLength=44100*2, displayAmplitude=5,
        autoscale=False, initialDirectory='', initialBaseFileName='',
        showFileWidgets=True, **kwargs):
        ttk.LabelFrame.__init__(self, *args, **kwargs)

        self.channels = []
        self.displayWidgets = {}
        self.historyLength = historyLength          # Max number of samples to display in history
        self.displayAmplitude = displayAmplitude    # Max amplitude to display (if autoscale=False)
        self.autoscale = autoscale                  # Autoscale axes
        self.audioTraces = []                        # matplotlib line
        self.showFileWidgets = showFileWidgets

        self.fileWidget = FileWritingEntry(
            self,
            defaultDirectory=initialDirectory,
            defaultBaseFileName=initialBaseFileName,
            purposeText='audio writing',
            text="Audio Writing"
            )

        self.enableViewerVar = tk.BooleanVar(); self.enableViewerVar.set(True); self.enableViewerVar.trace('w', self.updateEnableViewerCheckButton)
        self.enableViewerCheckButton = tk.Checkbutton(self, text="Enable viewer", variable=self.enableViewerVar, offvalue=False, onvalue=True)
        self.updateEnableViewerCheckButton()

        self.enableWriteChangeHandler = lambda:None
        self.enableWriteVar = tk.BooleanVar(); self.enableWriteVar.set(True); self.enableWriteVar.trace('w', self.updateEnableWriteCheckButton)
        self.enableWriteCheckButton = tk.Checkbutton(self, text="Enable write", variable=self.enableWriteVar, offvalue=False, onvalue=True)
        self.updateEnableWriteCheckButton()

        self.masterDisplayFrame = ttk.Frame(self)

        self.data = None

        for index, channel in enumerate(self.channels):
            self.createChannelDisplay(channel, index)

        self.updateWidgets()

    def updateEnableWriteCheckButton(self, *args):
        self.enableWriteChangeHandler()
        if self.getEnableWrite():
            self.enableWriteCheckButton["fg"] = 'green'
        else:
            self.enableWriteCheckButton["fg"] = 'red'

    def viewerEnabled(self):
        return self.enableViewerVar.get()

    def updateEnableViewerCheckButton(self, *args):
        if self.viewerEnabled():
            self.enableViewerCheckButton["fg"] = 'green'
        else:
            self.enableViewerCheckButton["fg"] = 'red'

    def getDirectory(self):
        return self.fileWidget.getDirectory()

    def getBaseFileName(self):
        return self.fileWidget.getBaseFileName()

    def getEnableWrite(self):
        return self.enableWriteVar.get()

    def setEnableWriteChangeHandler(self, function):
        self.enableWriteChangeHandler = function

    def setDirectoryChangeHandler(self, function):
        self.fileWidget.setDirectoryChangeHandler(function)

    def setBaseFileNameChangeHandler(self, function):
        self.fileWidget.setBaseFileNameChangeHandler(function)

    def addAudioData(self, newData):
        # Concatenate new audio data with old data, trim to monitor length if
        #   necessary, and update displays
        if self.viewerEnabled():
            if newData.shape[1] > self.historyLength:
                # Ok, this is really unlikely, but just in case
                newData = newData[:, -self.historyLength:]

            if self.data is None or self.data.shape[0] != newData.shape[0]:
                # Either data is uninitialied or the new data has a different # of channels from the old data)
                #   Note that a change in # of channels isn't really supported, just trying to avoid crashing
                self.data = np.empty((newData.shape[0], 0), dtype=newData.dtype)

            # Pad data to ensure it's self.historyLength long
            padAmount = self.historyLength - self.data.shape[1]
            if padAmount > 0:
                # Pad data up to desired historyLength
                self.data = np.pad(self.data, [(0, 0), (padAmount, 0)])
            elif padAmount < 0:
                # Data is too long for some reason
                self.data = self.data[:, -self.historyLength:]

            # Now data is guaranteed to have shape (N x L), N=# of channels, L=self.historyLength
            self.data = np.roll(self.data, -newData.shape[1], axis=1)
            self.data[:, -newData.shape[1]:] = newData

            # Display new data
            for k, channel in enumerate(self.channels):
                axes = self.displayWidgets[channel]['axes']
                fig = self.displayWidgets[channel]['figure']
                if len(self.audioTraces) < k+1 or self.data[k, :].shape != self.audioTraces[k].get_ydata().shape:
                    # Either there is no audio data, or audio data has changed shape, so we clear axes and redraw from scratch
                    axes.clear()
                    self.audioTraces.append(axes.plot(self.data[k, :].tolist(), LINE_STYLES[k % len(LINE_STYLES)], linewidth=1)[0])
                else:
                    # Audio data already exists, and has not changed shape, just update it.
                    self.audioTraces[k].set_ydata(self.data[k, :])

                if self.autoscale:
                    axes.relim()
                    axes.autoscale_view(True, True, True)
                else:
                    axes.set_ylim([-self.displayAmplitude, self.displayAmplitude])
                axes.margins(x=0, y=0)
                fig.canvas.draw()
                fig.canvas.flush_events()

    def updateChannels(self, channels):
        self.channels = channels
        self.updateWidgets()

    def updateWidgets(self):
        oldChannels = list(self.displayWidgets.keys())
        if not all(chan in oldChannels for chan in self.channels) or \
           not all(chan in self.channels for chan in oldChannels):
            # We've got a new set of channels, delete old widgets, make new ones
            for channel in self.displayWidgets:
                self.displayWidgets[channel]['displayFrame'].grid_forget()
                self.displayWidgets[channel]['displayFrame'].destroy()
                self.displayWidgets[channel]['figureCanvas'].get_tk_widget().pack_forget()
                self.displayWidgets[channel]['figureCanvas'].get_tk_widget().destroy()
                # Memory leak? MPL figure is probably still in memory...
                self.data = None
            self.displayWidgets = {}
            for index, channel in enumerate(self.channels):
                self.createChannelDisplay(channel, index)

#        wA, hA = getOptimalMonitorGrid(len(self.channels))
        for k, channel in enumerate(self.channels):
            self.displayWidgets[channel]['displayFrame'].pack()
            self.displayWidgets[channel]['figureCanvas'].get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        if len(self.channels) > 0:
            # No channels, it would look weird to display directory entry
            self.masterDisplayFrame.grid(row=0, column=0, columnspan=2)
            if self.showFileWidgets:
                self.fileWidget.grid(row=1, column=0, rowspan=2, sticky=tk.NSEW)
                self.enableViewerCheckButton.grid(row=1, column=1)
                self.enableWriteCheckButton.grid(row=2, column=1)
            else:
                self.fileWidget.grid_remove()
                self.enableViewerCheckButton.grid_remove()
                self.enableWriteCheckButton.grid_remove()
        else:
            self.masterDisplayFrame.grid_forget()
            self.fileWidget.grid_forget()
            self.enableViewerCheckButton.grid_remove()
            self.enableWriteCheckButton.grid_remove()

    def createChannelDisplay(self, channel, index):
        self.displayWidgets[channel] = {}  # Change this to gracefully remove existing channel widgets under this channel name
        self.displayWidgets[channel]['displayFrame'] = ttk.LabelFrame(self.masterDisplayFrame, text=channel)
        fig = Figure(figsize=(7, 0.75), dpi=100, facecolor=WIDGET_COLORS[1])
        t = np.arange(self.historyLength)
        axes = fig.add_subplot(111)
        axes.autoscale(enable=True)
        axes.plot(t, 0 * t, LINE_STYLES[index % len(LINE_STYLES)], linewidth=1)
        axes.relim()
        axes.autoscale_view(True, True, True)
        axes.margins(x=0, y=0)
        #fig.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        canvas = FigureCanvasTkAgg(fig, master=self.displayWidgets[channel]['displayFrame'])  # A tk.DrawingArea.
        canvas.draw()

        # Set up matplotlib figure callbacks
        # toolbar = NavigationToolbar2Tk(canvas, self.audioMonitorFrames[channel])
        # toolbar.update()
#         def figureKeyPressManager(event):
# #            syncPrint("you pressed {}".format(event.key))
#             key_press_handler(event, canvas, toolbar)
#         canvas.mpl_connect("key_press_event", figureKeyPressManager)

        self.displayWidgets[channel]['figure'] = fig
        self.displayWidgets[channel]['axes'] = axes
        self.displayWidgets[channel]['figureCanvas'] = canvas
        # self.displayWidgets[channel]['figureNavToolbar'] = toolbar
        # self.displayWidgets[channel]['figureLine'] = line

class CameraMonitor(ttk.LabelFrame):
    def __init__(self, *args, displaySize=(400, 300),
                    camSerial='Unknown camera', speedText='Unknown speed',
                    initialDirectory='', initialBaseFileName='',
                    showFileWidgets=True, **kwargs):
        ttk.LabelFrame.__init__(self, *args, **kwargs)
        self.camSerial = camSerial
        self.config(text="{serial} ({speed})".format(serial=self.camSerial, speed=speedText))
        self.displaySize = displaySize
        self.canvas = tk.Canvas(self, width=self.displaySize[0], height=self.displaySize[1], borderwidth=2, relief=tk.SUNKEN)
        self.imageID = None
        self.currentImage = None
        self.showFileWidgets = showFileWidgets

        self.isIdle = False  # Boolean flag indicating whether the monitor is actively sending images or not

        self.fileWidget = FileWritingEntry(
            self,
            defaultDirectory=initialDirectory,
            defaultBaseFileName=initialBaseFileName,
            purposeText='video writing',
            text="Video Writing - {camSerial}".format(camSerial=self.camSerial)
            )

        self.enableViewerVar = tk.BooleanVar(); self.enableViewerVar.set(True); self.enableViewerVar.trace('w', self.updateEnableViewerCheckButton)
        self.enableViewerCheckButton = tk.Checkbutton(self, text="Enable viewer", variable=self.enableViewerVar, offvalue=False, onvalue=True)
        self.updateEnableViewerCheckButton()

        self.enableWriteChangeHandler = lambda:None
        self.enableWriteVar = tk.BooleanVar(); self.enableWriteVar.set(True); self.enableWriteVar.trace('w', self.updateEnableWriteCheckButton)
        self.enableWriteCheckButton = tk.Checkbutton(self, text="Enable write", variable=self.enableWriteVar, offvalue=False, onvalue=True)
        self.updateEnableWriteCheckButton()

        self.canvas.grid(row=0, column=0, columnspan=2)
        if self.showFileWidgets:
            self.fileWidget.grid(row=1, column=0, rowspan=2, sticky=tk.NSEW)
            self.enableViewerCheckButton.grid(row=1, column=1)
            self.enableWriteCheckButton.grid(row=2, column=1)
        else:
            self.fileWidget.grid_remove()
            self.enableViewerCheckButton.grid_remove()
            self.enableWriteCheckButton.grid_remove()

        # Initialize widget with idle image
        self.idle()

    def getDisplaySize(self):
        return self.displaySize
    def setDisplaySize(self, newSize):
        self.displaySize = newSize
        self.canvas['width'] = self.displaySize[0]
        self.canvas['height'] = self.displaySize[1]

    def idle(self):
        if not self.isIdle:
            # Transitioning from active to idle
            self.isIdle = True
            self.updateImage(NO_IMAGES_IMAGE)
    def active(self):
        self.isIdle = False

    def updateEnableWriteCheckButton(self, *args):
        self.enableWriteChangeHandler()
        if self.getEnableWrite():
            self.enableWriteCheckButton["fg"] = 'green'
        else:
            self.enableWriteCheckButton["fg"] = 'red'

    def viewerEnabled(self):
        return self.enableViewerVar.get()
    def updateEnableViewerCheckButton(self, *args):
        if self.viewerEnabled():
            self.enableViewerCheckButton["fg"] = 'green'
        else:
            self.enableViewerCheckButton["fg"] = 'red'

    def createAttributeBrowserNode(self, attributeNode, parent, tooltipLabel, gridRow):
        frame = ttk.Frame(parent)
        frame.bind("<Enter>", lambda event: tooltipLabel.config(text=attributeNode["tooltip"]))  # Set tooltip rollover callback
        frame.grid(row=gridRow)

        # syncPrint()
        # pp = pprint.PrettyPrinter(indent=1, depth=1)
        # pp.pprint(attributeNode)
        # syncPrint()

        widgets = [frame]
        childWidgets = []
        childCategoryHolder = None
        childCategoryWidgets = []

        if attributeNode['type'] == "category":
            children = []
            parent.add(frame, text=attributeNode['displayName'])
            if len(attributeNode['subcategories']) > 0:
                # If this category has subcategories, create a notebook to hold them
                childCategoryHolder = ttk.Notebook(frame)
                childCategoryHolder.grid(row=0)
                widgets.append(childCategoryHolder)
                for subcategoryAttributeNode in attributeNode['subcategories']:
                    childCategoryWidgets.append(self.createAttributeBrowserNode(subcategoryAttributeNode, childCategoryHolder, tooltipLabel, 0))
            for k, childAttributeNode in enumerate(attributeNode['children']):
                childWidgets.append(self.createAttributeBrowserNode(childAttributeNode, frame, tooltipLabel, k+1))
        else:
            if attributeNode['accessMode'] == "RW":
                # Read/write attribute
                accessState = 'normal'
            else:
                # Read only attribute
                accessState = 'readonly'
            if attributeNode['type'] == "command":
                commandButton = ttk.Button(frame, text=attributeNode['displayName'])
                commandButton.grid()
                widgets.append(commandButton)
            elif attributeNode['type'] == "enum":
                enumLabel = ttk.Label(frame, text=attributeNode['displayName'])
                enumLabel.grid(column=0, row=0)
                options = list(attributeNode['options'].values())
                enumSelector = ttk.Combobox(frame, state=accessState, values=options)
                enumSelector.set(attributeNode['value'][1])
                enumSelector.grid(column=1, row=0)
                widgets.append(enumLabel)
                widgets.append(enumSelector)
            else:
                entryLabel = ttk.Label(frame, text=attributeNode['displayName'])
                entryLabel.grid(column=0, row=0)
                entry = ttk.Entry(frame, state=accessState)
                entry.insert(0, attributeNode['value'])
                entry.grid(column=1, row=0)
                widgets.append(entryLabel)
                widgets.append(entry)

        return {'widgets':widgets, 'childWidgets':childWidgets, 'childCategoryWidgets':childCategoryWidgets, 'childCategoryHolder':childCategoryHolder}

    def setEnableWriteChangeHandler(self, function):
        self.enableWriteChangeHandler = function

    def setDirectoryChangeHandler(self, function):
        self.fileWidget.setDirectoryChangeHandler(function)

    def setBaseFileNameChangeHandler(self, function):
        self.fileWidget.setBaseFileNameChangeHandler(function)

    def updateImage(self, image, pixelFormat=None):
        # Expects a PIL image object
        if self.viewerEnabled():
            if cu.pixelFormats[pixelFormat]['bayer']:
                # Invert bayer filter to get full color image
                image = Image.fromarray(cv2.cvtColor(np.asarray(image), cv2.COLOR_BayerRGGB2RGB))
            newSize = self.getBestImageSize(image.size)
            image = image.resize(newSize, resample=Image.BILINEAR)
            self.currentImage = ImageTk.PhotoImage(image)
            if self.imageID is None:
                self.imageID = self.canvas.create_image((0, 0), image=self.currentImage, anchor=tk.NW)
            else:
                self.canvas.itemconfig(self.imageID, image=self.currentImage)

    def getBestImageSize(self, imageSize):
        # Get a new image size that preserves the aspect ratio, and fits into
        #   the display canvas without cutting off any image or wasting space.
        xRatio = self.displaySize[0] / imageSize[0]
        yRatio = self.displaySize[1] / imageSize[1]

        if xRatio > yRatio:
            # image aspect ratio is wider than display - scale based on x ratio
            return (self.displaySize[0], int(imageSize[1] * xRatio))
        else:
            # image aspect ratio is taller than display - scale based on y ratio
            return (int(imageSize[0] * yRatio), self.displaySize[1])

    def getDirectory(self):
        return self.fileWidget.getDirectory()

    def getBaseFileName(self):
        return self.fileWidget.getBaseFileName()

    def getEnableWrite(self):
        return self.enableWriteVar.get()

    def destroy(self):
        ttk.LabelFrame.destroy(self)
        # self.fileWidget.grid_forget()
        # self.canvas.grid_forget()
        self.imageID = None
        self.currentImage = None
        # self.cameraAttributeBrowserButton.grid_forget()
