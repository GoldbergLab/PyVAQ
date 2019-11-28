import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
from PIL import Image, ImageTk
from SharedImageQueue import SharedImageSender
from scipy.signal import butter, lfilter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
# from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from fileWritingEntry import FileWritingEntry

WIDGET_COLORS = [
    '#050505', # near black
    '#e6f5ff', # very light blue
    '#c1ffc1', # light green
    '#FFC1C1'  # light red
]
LINE_STYLES = [c+'-' for c in 'bykcmgr']

class AudioMonitor(ttk.LabelFrame):
    def __init__(self, *args, historyLength=44100*2, displayAmplitude=5, autoscale=False, **kwargs):
        ttk.LabelFrame.__init__(self, *args, **kwargs)

        self.channels = []
        self.displayWidgets = {}
        self.historyLength = historyLength          # Max number of samples to display in history
        self.displayAmplitude = displayAmplitude    # Max amplitude to display (if autoscale=False)
        self.autoscale = autoscale                  # Autoscale axes

        self.fileWidget = FileWritingEntry(
            self,
            defaultDirectory=r'C:\Users\Brian Kardon\Documents\Cornell Lab Tech non-syncing\PyVAQ test videos\audio',
            defaultBaseFileName='audioWrite',
            purposeText='audio writing',
            text="Audio Writing"
            )

        self.masterDisplayFrame = ttk.Frame(self)

        self.data = None

        for index, channel in enumerate(self.channels):
            self.createChannelDisplay(channel, index)

        self.updateWidgets()

    def getDirectory(self):
        return self.fileWidget.getDirectory()

    def getBaseFileName(self):
        return self.fileWidget.getBaseFileName()

    def setDirectoryChangeHandler(self, function):
        self.fileWidget.setDirectoryChangeHandler(function)

    def setBaseFileNameChangeHandler(self, function):
        self.fileWidget.setBaseFileNameChangeHandler(function)

    def addAudioData(self, newData):
        # Concatenate new audio data with old data, trim to monitor length if
        #   necessary, and update displays
        if self.data is None:
            self.data = newData
        else:
            self.data = np.concatenate((self.data, newData), axis=1)

        if self.data is not None:
            # Trim data to specified size
            startTrim = self.data.shape[1] - self.historyLength
            if startTrim < 0:
                startTrim = 0
            self.data = self.data[:, startTrim:]

            # Display new data
            for k, channel in enumerate(self.channels):
                axes = self.displayWidgets[channel]['axes']
                fig = self.displayWidgets[channel]['figure']
                axes.clear()
                axes.plot(self.data[k, :].tolist(), LINE_STYLES[k % len(LINE_STYLES)], linewidth=1)
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
        if self.channels != self.displayWidgets.keys():
            # We've got a new set of channels, delete old widgets, make new ones
            for channel in self.displayWidgets:
                self.displayWidgets[channel]['displayFrame'].grid_forget()
                self.displayWidgets[channel]['figureCanvas'].pack_forget()
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
            self.masterDisplayFrame.grid(row=0, column=0)
            self.fileWidget.grid(row=1, column=0)
        else:
            self.masterDisplayFrame.grid_forget()
            self.fileWidget.grid_forget()


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
    def __init__(self, *args, displaySize=(400, 300), camSerial='Unknown camera', speedText='Unknown speed', **kwargs):
        ttk.LabelFrame.__init__(self, *args, **kwargs)
        self.camSerial = camSerial
        self.config(text="{serial} ({speed})".format(serial=self.camSerial, speed=speedText))
        self.displaySize = displaySize
        self.canvas = tk.Canvas(self, width=self.displaySize[0], height=self.displaySize[1], borderwidth=2, relief=tk.SUNKEN)
        self.imageID = None
        self.currentImage = None

        self.fileWidget = FileWritingEntry(
            self,
            defaultDirectory=r'C:\Users\Brian Kardon\Documents\Cornell Lab Tech non-syncing\PyVAQ test videos\video',
            defaultBaseFileName='videoWrite',
            purposeText='video writing',
            text="Video Writing - {camSerial}".format(camSerial=self.camSerial)
            )

        self.canvas.grid(row=0, column=0)
        self.fileWidget.grid(row=1, column=0, sticky=tk.NSEW)

#       self.cameraAttributeBrowserButton = ttk.Button(vFrame, text="Attribute browser", command=lambda:self.createCameraAttributeBrowser(camSerial))

    def createCameraAttributeBrowser(self, camSerial):
        main = tk.Toplevel()
        nb = ttk.Notebook(main)
        nb.grid(row=0)
        tooltipLabel = ttk.Label(main, text="temp")
        tooltipLabel.grid(row=1)

        #self.cameraAttributesWidget[camSerial]
        widgets = self.createAttributeBrowserNode(self.cameraAttributes[camSerial], nb, tooltipLabel, 1)

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

    def setDirectoryChangeHandler(self, function):
        self.fileWidget.setDirectoryChangeHandler(function)

    def setBaseFileNameChangeHandler(self, function):
        self.fileWidget.setBaseFileNameChangeHandler(function)

    def updateImage(self, image):
        # Expects a PIL image object
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

    def destroy(self):
        ttk.LabelFrame.destroy(self)
        # self.fileWidget.grid_forget()
        # self.canvas.grid_forget()
        self.imageID = None
        self.currentImage = None
        # self.cameraAttributeBrowserButton.grid_forget()
