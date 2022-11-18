import tkinter as tk
import tkinter.ttk as ttk
import PySpinUtilities as psu

class CameraConfigPanel(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.camSerials = []
        self.storedAttributes = {}

        self.filterLabel = tk.Label(self, text='Filter')
        self.filterVar = tk.StringVar()
        self.filterEntry = tk.Entry(self, textvariable=self.filterVar)

        self.reloadButton = tk.Button(self, text="Reload settings", command=self.updateCameraAttributes)

        self.cameraLabel = tk.Label(self, text="Camera")
        self.cameraVar = tk.StringVar()
        self.cameraList = ttk.Combobox(self, textvariable=self.cameraVar)

        self.attributeLabel = tk.Label(self, text="Setting")
        self.attributeVar = tk.StringVar()
        self.attributeList = ttk.Combobox(self, textvariable=self.attributeVar)

        self.valueLabel = tk.Label(self, text="Value")
        self.valueVar = tk.StringVar()
        # Either the value entry or the value list will be used depending on which type of attribute is selected
        self.valueEntry = tk.Entry(self, textvariable=self.valueVar)
        self.valueList = ttk.Combobox(self, textvariable=self.valueVar)

        self.restoreButton = tk.Button(self, text="Restore current value", command=self.restoreCurrentValue)
        self.applyButton = tk.Button(self, text="Apply setting", command=self.applySetting)

        self.categoryLabel = tk.Label(self)

        self.filterLabel.grid(row=1, column=0)
        self.filterEntry.grid(row=1, column=1)
        self.reloadButton.grid(row=1, column=2)
        self.cameraLabel.grid(row=2, column=0)
        self.cameraList.grid(row=3, column=0)
        self.attributeLabel.grid(row=2, column=1)
        self.attributeList.grid(row=3, column=1)
        self.valueLabel.grid(row=2, column=2, columnspan=2)
        self.valueEntry.grid(row=3, column=2, columnspan=2)
        self.valueList.grid(row=3, column=2, columnspan=2)
        self.restoreButton.grid(row=4, column=2)
        self.applyButton.grid(row=4, column=3)

        self.updateCameraList()
        self.updateCameraAttributes()

        self.grid()

    def updateCameraList(self):
        """Rediscover what cameras are available, update camera dropdown.

        Returns:
            None.

        """

        self.camSerials = psu.discoverCameras()
        self.cameraList['values'] = self.camSerials
        self.cameraList.current(0)

    def getCurrentCamSerial(self):
        idx = self.cameraList.current()
        if idx == -1:
            return None
        else:
            return self.cameraList['values'][idx]

    def updateCameraAttributes(self):
        """Reload current attributes from camera, update widgets.

        Returns:
            type: Description of returned object.

        """

        camSerial = self.getCurrentCamSerial()
        if camSerial is None:
            # No camera selected
            return

        nestedAttributes = psu.getAllCameraAttributes(camSerial=camSerial)
        flattenedAttributes = psu.flattenCameraAttributes(nestedAttributes)
        self.storedAttributes[camSerial] = flattenedAttributes
        self.updateAttributes()
        self.updateValue()
        self.updateApplyButtonState()
        self.updateCategoryList()

    def updateAttributes(self):
        """Use stored attributes to update widgets for current camera.

        If no stored attributes exist for the currently selected camera, then
            load them from the camera

        Returns:
            None

        """

        camSerial = self.getCurrentCamSerial()
        if camSerial is not None:
            attributeNames = [attribute['displayName'] for attribute in self.storedAttributes[camSerial]]
            self.attributeList['values'] = attributeNames
            if len(attributeNames) > 0:
                self.attributeVar.set(attributeNames[0])

    def updateValue(self):
        """Update value display to reflect the current stored value

        Also switch from entry to combobox depending on whether the selected
            attribute is an enum or not

        Returns:
            type: Description of returned object.

        """

        attribute = self.getAttribute()
        if attribute is None:
            self.valueVar.set('')
            return
        if attribute['type'] == 'enum':
            self.valueList['values'] = attribute['options']
            self.valueEntry.grid_remove()
            self.valueList.grid()
        else:
            self.valueList.grid_remove()
            self.valueEntry.grid()
        self.valueVar.set(attribute['value'])

    def getAttribute(self, displayName=None):
        camSerial = self.getCurrentCamSerial()
        if camSerial is None:
            return None
        if displayName is None:
            displayName = self.attributeVar.get()
        for attribute in self.storedAttributes[camSerial]:
            if attribute['displayName'] == displayName:
                return attribute
        return None

    def updateApplyButtonState(self):
        """Disable/enable the "apply" button according to attribute read-only state.

        Returns:
            None

        """

        attribute = self.getAttribute()
        if attribute is None or attribute['accessMode'] == 'RO':
            # Attribute is read only - disable apply button
            self.applyButton['state'] = tk.DISABLED
        elif attribute['accessMode'] == 'RW':
            # Attribute is read/write - enable apply button
            self.applyButton['state'] = tk.NORMAL

    def applySetting(self):
        """Attempt to apply the setting to the camera.

        Also reload camera settings afterward.

        Returns:
            None

        """

        attribute = self.getAttribute()
        attributeName = attribute['name']
        attributeValue = self.valueVar.get()
        attributeType = attribute['type']
        psu.setCameraAttribute(attributeName, attributeValue, camSerial=self.cameraVar.get(), type=attributeType, nodemap='NodeMap')

        self.updateCameraAttributes()

    def restoreCurrentValue(self):
        """Restore stored value for the selected attribute.

        Returns:
            None

        """
        self.updateValue()

    def updateCategoryList(self):
        """Update the category path for the selected attribute.

        Returns:
            None

        """

        attribute = self.getAttribute()
        path = ' > '.join(attribute['path'])
        self.categoryLabel['text'] = path
