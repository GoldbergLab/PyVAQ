import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
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
import PySpinUtilities as psu
from CollapsableFrame import CollapsableFrame

WIDGET_COLORS = [
    '#050505', # near black
    '#e6f5ff', # very light blue
    '#c1ffc1', # light green
    '#FFC1C1'  # light red
]
LINE_STYLES = [c+'-' for c in 'bykcmgr']

with Image.open(r'Resources\NoImages_000.png') as NO_IMAGES_IMAGE:
    NO_IMAGES_IMAGE.load()

class BaseMonitor(CollapsableFrame):
    """A base class that other monitor classes inherit from.

    This defines a GUI widget for viewing a data stream. It has subwidgets for
        specifying where/how to write data files, for enabling/disabling the
        viewer

    Args:
        *args (*): Arguments to pass to the CollapsableFrame parent class.
        initialDirectory (str): Path to a directory to save data files to.
            Defaults to ''.
        initialBaseFileName (str): Base file name to save data files to.
            Defaults to ''.
        showFileWidgets (bool): A boolean flag indicating whether or not the
            file widgets should be displayed. Defaults to True.
        filePurposeText (str): A string that will be displayed to the user when
            they open a window to select a new directory to save data files to.
            Defaults to 'file writing'.
        fileText (type): A string that will be displayed to the user as the
            title of a window which they can open to select a new directory to
            save data files to.
        **kwargs (**): Keyword arguments to pass to the CollapsableFrame parent
            class.

    Attributes:
        fileWidget (FileWritingEntry): A sub-widget to allow the user to specify
            a base filename and directory for saving data files.
        showFileWidgets

    """
    def __init__(self, *args, initialDirectory='', initialBaseFileName='',
        showFileWidgets=True, filePurposeText='file writing',
        fileText='File Writing', **kwargs):
        # Initialize parent class
        CollapsableFrame.__init__(self, *args, **kwargs)

        self.showFileWidgets = showFileWidgets

        # Create file writing widget
        self.fileWidget = FileWritingEntry(
            self,
            defaultDirectory=initialDirectory,
            defaultBaseFileName=initialBaseFileName,
            purposeText=filePurposeText,
            text=fileText
            )

        # Set up the tkinter variable to track whether the viewer is enabled or not
        self.enableViewerVar = tk.BooleanVar(); self.enableViewerVar.set(True); self.enableViewerVar.trace('w', self.updateEnableViewerCheckButton)
        self.enableViewerCheckButton = tk.Checkbutton(self, text="Enable viewer", variable=self.enableViewerVar, offvalue=False, onvalue=True)
        self.updateEnableViewerCheckButton()

        # Set up the tkinter variable to track whether file writing is enabled or not
        self.enableWriteChangeHandler = lambda:None
        self.enableWriteVar = tk.BooleanVar(); self.enableWriteVar.set(True); self.enableWriteVar.trace('w', self.updateEnableWriteCheckButton)
        self.enableWriteCheckButton = tk.Checkbutton(self, text="Enable write", variable=self.enableWriteVar, offvalue=False, onvalue=True)
        self.updateEnableWriteCheckButton()

        # Create the main inner container for data monitor widgets
        self.mainDisplayFrame = ttk.Frame(self)

        self.collapseFunction = self.disableViewer
        self.expandFunction = self.enableViewer

    def updateEnableWriteCheckButton(self, *args):
        """Update the appearance of the enable write checkbox.

        Args:
            *args (*): Unused

        Returns:
            None

        """

        # Run any callbacks the user has requested when the state changes
        self.enableWriteChangeHandler()
        # Set the checkbox color based on the enable write state
        if self.getEnableWrite():
            self.enableWriteCheckButton["fg"] = 'green'
        else:
            self.enableWriteCheckButton["fg"] = 'red'

    def viewerEnabled(self):
        return self.enableViewerVar.get()

    def enableViewer(self, *args, **kwargs):
        self.enableViewerVar.set(True)

    def disableViewer(self, *args, **kwargs):
        self.enableViewerVar.set(False)

    def toggleViewerEnable(self):
        self.enableViewerVar.set(not self.enableViewerVar.get())

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

class DigitalMonitor(BaseMonitor):
    def __init__(self, *args, historyLength=44100*2, **kwargs):
        BaseMonitor.__init__(self, *args, filePurposeText='digital writing',
            fileText='Digital Writing', text="Digital Monitor", collapsed=False,
            **kwargs)
        self.channels = []
        self.historyLength = historyLength          # Max number of samples to display in history

        self.displayWidth = 600
        self.displayHeight = 200
        self.canvas = tk.Canvas(self.mainDisplayFrame, width=self.displayWidth, height=self.displayHeight)

        self.currentImage = None
        self.imageID = None

        self.data = None

        self.viewerChannelCount = None
        self.channelHeight = 20
        self.viewWidth = 600
        self.stripes = None
        self.labelImage = None

        self.updateWidgets()

    def addDigitalData(self, newData):
        if self.viewerEnabled():
            # newData = np.random.rand(*newData.shape) > 0.8

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

            self.updateDataImage()

    def updateDataImage(self):
        dataImageArray = (np.repeat(np.expand_dims(self.data, 2), 3, axis=2) * 200).astype('uint8')

        if self.viewerChannelCount is None or self.viewerChannelCount != self.data.shape[0]:
            self.viewerChannelCount = self.data.shape[0]

            self.canvas.config(height=self.viewerChannelCount * self.channelHeight)

            # (Re)create the striping pattern and the channel numbers

            stripeIntensity = 50
            self.stripes = np.expand_dims(((np.array(range(self.viewerChannelCount)) % 2) * stripeIntensity).astype('uint8'), [1, 2])
            self.stripes = np.concatenate((self.stripes*0, self.stripes*0, self.stripes), axis=2)

            font = ImageFont.truetype('.\Resources\segoeuib.ttf', int(self.channelHeight * 0.7))

            self.labelImage = Image.new('RGBA', (self.viewWidth, self.viewerChannelCount * self.channelHeight))
            labelDraw = ImageDraw.Draw(self.labelImage)
            for k in range(self.viewerChannelCount):
                labelDraw.text((int(self.channelHeight/4), k*self.channelHeight), str(k), font=font, fill='#ff0000')

        dataImageArray = dataImageArray + self.stripes

        dataImage = Image.fromarray(dataImageArray, mode='RGB').resize((self.viewWidth, self.viewerChannelCount*self.channelHeight), resample=Image.NEAREST)
        dataImage.paste(self.labelImage, mask=self.labelImage)

        self.currentImage = ImageTk.PhotoImage(dataImage)
        if self.imageID is None:
            self.imageID = self.canvas.create_image((0, 0), image=self.currentImage, anchor=tk.NW)
        else:
            self.canvas.itemconfig(self.imageID, image=self.currentImage)

    def updateChannels(self, channels):
        self.channels = channels
        self.updateWidgets()

    def updateWidgets(self):

        if len(self.channels) > 0:
            # No channels, it would look weird to display directory entry
            self.mainDisplayFrame.grid(row=0, column=0, columnspan=2, sticky=tk.NSEW)
            self.canvas.grid(row=0, column=0)

            if self.showFileWidgets:
                self.fileWidget.grid(row=1, column=0, rowspan=2, sticky=tk.NSEW)
                self.enableViewerCheckButton.grid(row=1, column=1)
                self.enableWriteCheckButton.grid(row=2, column=1)
            else:
                self.fileWidget.grid_remove()
                self.enableViewerCheckButton.grid_remove()
                self.enableWriteCheckButton.grid_remove()
        else:
            self.mainDisplayFrame.grid_forget()
            self.fileWidget.grid_forget()
            self.enableViewerCheckButton.grid_remove()
            self.enableWriteCheckButton.grid_remove()

class AudioMonitor(BaseMonitor):
    def __init__(self, *args, historyLength=44100*2, displayAmplitude=5,
        autoscale=False, **kwargs):
        BaseMonitor.__init__(self, *args, filePurposeText='audio writing',
            fileText='Audio Writing', text="Audio Monitor", collapsed=False,
            **kwargs)

        self.channels = []
        self.displayWidgets = {}
        self.historyLength = historyLength          # Max number of samples to display in history
        self.displayAmplitude = displayAmplitude    # Max amplitude to display (if autoscale=False)
        self.autoscale = autoscale                  # Autoscale axes
        self.audioTraces = []                        # matplotlib line

        self.data = None

        for index, channel in enumerate(self.channels):
            self.createChannelDisplay(channel, index)

        self.updateWidgets()

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
            self.displayWidgets[channel]['displayFrame'].pack(anchor=tk.NW, fill=tk.X, expand=True)
            self.displayWidgets[channel]['figureCanvas'].get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        if len(self.channels) > 0:
            # No channels, it would look weird to display directory entry
            self.mainDisplayFrame.grid(row=0, column=0, columnspan=2, sticky=tk.NSEW)
            if self.showFileWidgets:
                self.fileWidget.grid(row=1, column=0, rowspan=2, sticky=tk.NSEW)
                self.enableViewerCheckButton.grid(row=1, column=1)
                self.enableWriteCheckButton.grid(row=2, column=1)
            else:
                self.fileWidget.grid_remove()
                self.enableViewerCheckButton.grid_remove()
                self.enableWriteCheckButton.grid_remove()
        else:
            self.mainDisplayFrame.grid_forget()
            self.fileWidget.grid_forget()
            self.enableViewerCheckButton.grid_remove()
            self.enableWriteCheckButton.grid_remove()

    def createChannelDisplay(self, channel, index, collapsable=True):
        if collapsable:
            frameType = CollapsableFrame
        else:
            frameType = ttk.LabelFrame
        self.displayWidgets[channel] = {}  # Change this to gracefully remove existing channel widgets under this channel name
        self.displayWidgets[channel]['displayFrame'] = frameType(self.mainDisplayFrame, text=channel)
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

class CameraMonitor(BaseMonitor):
    def __init__(self, *args, displaySize=(400, 300),
                    camSerial='Unknown camera', speedText='Unknown speed',
                    text="Camera Monitor", collapsed=False, **kwargs):
        self.camSerial = camSerial
        fileText = "Video Writing - {camSerial}".format(camSerial=self.camSerial)
        BaseMonitor.__init__(self, *args, filePurposeText='video writing',
            fileText=fileText, **kwargs)
        self.setText("{serial} ({speed})".format(serial=self.camSerial, speed=speedText))
        self.displaySize = displaySize
        self.canvas = tk.Canvas(self, width=self.displaySize[0], height=self.displaySize[1], borderwidth=2, relief=tk.SUNKEN)
        self.imageID = None
        self.currentImage = None

        self.isIdle = False  # Boolean flag indicating whether the monitor is actively sending images or not

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

    def updateImage(self, image, pixelFormat=None):
        # Expects a PIL image object
        if self.viewerEnabled():
            if psu.pixelFormats[pixelFormat]['bayer']:
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

    def destroy(self):
        ttk.LabelFrame.destroy(self)
        # self.fileWidget.grid_forget()
        # self.canvas.grid_forget()
        self.imageID = None
        self.currentImage = None
        # self.cameraAttributeBrowserButton.grid_forget()
