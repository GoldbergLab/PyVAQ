import tkinter as tk
import tkinter.ttk as ttk
import PySpinUtilities as psu

class CameraConfigPanel(tk.Frame):
    """A tkinter widget allowing for FLIR camera configuration.

    Args:
        parent (widget): A container widget to be the CameraConfigPanel parent

    Attributes:
        camSerials (list of str): List of serial #s of available cameras
        storedAttributes (dict of lists): A dict with camera serials as keys
            containing a list of camera attributes
        filterLabel (tk.Label): Label for the filter entry
        filterVar (tk.StringVar): Variable containing filter text.
        filterEntry (tk.Entry): Entry widget for filtering attributes.
        reloadButton (tk.Button): Button for reloading attributes from camera
        cameraLabel (tk.Label): Label for camera combobox.
        cameraVar (tk.StringVar): Variable containing the currently selected
            camera serial.
        cameraList (ttk.Combobox): Dropdown list of currently available camera
            serials.
        attributeLabel (tk.Label): Label for the attribute entry.
        attributeVar (tk.StringVar): Variable containing the currently selected
            attribute
        attributeList (ttk.Combobox): Dropdown list of filtered attributes
        valueLabel (tk.Label): Label for value entry
        valueVar (tk.StringVar): Variable containing the current attribute
            value
        valueEntry (tk.Entry): Entry for displaying/changing the current
            attribute value
        valueList (ttk.Combobox): Dropdown list for displaying/selecting enum
            type attribute values
        applyButton (tk.Button): Apply the attribute value to the camera
        categoryLabel (tk.Entry): Label for displaying category path of
            currently selected attribute

    """
    def __init__(self, parent, configurationChangeHandler=lambda:None):
        super().__init__(parent)

        self.parent = parent
        self.camSerials = []
        self.storedAttributes = {}

        self.filterLabel = tk.Label(self, text='Attribute filter:', justify=tk.RIGHT)
        self.filterVar = tk.StringVar()
        self.filterVar.trace('w', self.handleFilterChange)
        self.filterEntry = tk.Entry(self, textvariable=self.filterVar)

        self.cameraLabel = tk.Label(self, text="Camera", justify=tk.LEFT)
        self.cameraVar = tk.StringVar()
        self.cameraList = ttk.Combobox(self, textvariable=self.cameraVar, exportselection=0)
        self.cameraList['state'] = 'readonly'

        self.attributeLabel = tk.Label(self, text="Attribute", justify=tk.LEFT)
        self.attributeVar = tk.StringVar()
        self.attributeVar.trace('w', self.handleAttributeChange)
        self.attributeList = ttk.Combobox(self, textvariable=self.attributeVar, exportselection=0, width=48)
        self.attributeList['state'] = 'readonly'

        self.valueLabel = tk.Label(self, text="Value", justify=tk.LEFT)
        self.valueVar = tk.StringVar()
        # Either the value entry or the value list will be used depending on which type of attribute is selected
        self.valueEntry = tk.Entry(self, textvariable=self.valueVar, width=32)
        self.valueList = ttk.Combobox(self, textvariable=self.valueVar, width=32, exportselection=0)
        self.valueList['state'] = 'readonly'

        self.buttonFrame = tk.Frame(self)

        self.reloadButton = tk.Button(self.buttonFrame, text="Reload attributes from camera", command=self.updateCameraAttributes)
#        self.restoreButton = tk.Button(self.buttonFrame, text="Restore current value", command=self.restoreCurrentValue)
        self.addButton = tk.Button(self.buttonFrame, text="\/\/ Add attribute to configuration \/\/", command=self.addCurrentAttributeToConfiguration)
        self.applyCurrentAttributeButton = tk.Button(self.buttonFrame, text="Apply this attribute now", command=self.applyCurrentAttribute)

        self.configurationEntry = tk.Entry(self, width=60)
        self.configurationEntry.insert(0, 'CameraSerial, AttributeName, AttributeValue, AttributeType')
        self.configurationEntry['state'] = tk.DISABLED
        self.configurationText = tk.Text(self, width=60, height=10)
        self.configurationChangeHandler = configurationChangeHandler
        self.configurationText.bind('<KeyRelease>', self.configurationChangeHandler)
        self.applyConfigurationButton = tk.Button(self, text="Apply configuration now", command=self.applyConfiguration)
        self.applyConfigurationOnInitVar = tk.BooleanVar()
        self.applyConfigurationOnInitVar.set(True)
        self.applyConfigurationOnInitCheckbox = ttk.Checkbutton(self, text='Apply configuration on initialization', variable=self.applyConfigurationOnInitVar, offvalue=False, onvalue=True)

        self.categoryVar = tk.StringVar()
        self.categoryLabel = tk.Entry(self, textvariable=self.categoryVar, width=48)
        self.categoryLabel['state'] = tk.DISABLED

        self.filterLabel.grid(row=1, column=0, sticky=tk.E)
        self.filterEntry.grid(row=1, column=1, sticky=tk.EW)
        self.categoryLabel.grid(row=1, column=2, columnspan=2, sticky=tk.EW)
        self.cameraLabel.grid(row=2, column=0, sticky=tk.W)
        self.cameraList.grid(row=3, column=0, sticky=tk.EW)
        self.attributeLabel.grid(row=2, column=1, sticky=tk.W)
        self.attributeList.grid(row=3, column=1)
        self.valueLabel.grid(row=2, column=2, columnspan=2, sticky=tk.W)
        self.valueEntry.grid(row=3, column=2, columnspan=2, sticky=tk.EW)
        self.valueList.grid(row=3, column=2, columnspan=2)
        self.buttonFrame.grid(row=4, column=0, columnspan=3, sticky=tk.E)
        self.reloadButton.grid(row=0, column=1, sticky=tk.E)
#        self.restoreButton.grid(row=0, column=2, sticky=tk.E)
        self.applyCurrentAttributeButton.grid(row=0, column=2, sticky=tk.E)
        self.addButton.grid(row=0, column=4, sticky=tk.E)

        self.configurationEntry.grid(row=5, column=0, columnspan=4, sticky=tk.NSEW)
        self.configurationText.grid(row=6, column=0, columnspan=4, sticky=tk.NSEW)
        self.applyConfigurationButton.grid(row=7, column=0, sticky=tk.EW)
        self.applyConfigurationOnInitCheckbox.grid(row=7, column=1, sticky=tk.E)

        self.updateCameraList()
        self.updateCameraAttributes()

        self.grid()

    def applyConfigurationOnInit(self):
        return self.applyConfigurationOnInitVar.get()

    def handleFilterChange(self, *args, **kwargs):
        """React to a change in the filter text.

        Update the attribute list by filtering

        Args:
            *args (type): Description of parameter `*args`.
            **kwargs (type): Description of parameter `**kwargs`.

        Returns:
            type: Description of returned object.

        """
        self.updateAttributes()

    def updateCameraList(self):
        """Rediscover what cameras are available, update camera dropdown.

        Returns:
            None.

        """

        self.camSerials = psu.discoverCameras()
        self.cameraList['values'] = self.camSerials
        self.cameraList.current(0)

    def getCurrentCamSerial(self):
        """Get the currently selected camera serial.

        Returns:
            str: The serial number of the selected camera

        """

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

    def checkFilter(self, attribute):
        """Check if an attribute matches the current filter text.

        Args:
            attribute (dict): A dict representing a camera attribute

        Returns:
            bool: A boolean representing whether or not the attribute matched

        """
        filterText = self.filterVar.get().lower()

        if len(filterText) == 0:
            return True

        if filterText in attribute['name'].lower():
            return True
        if filterText in attribute['displayName'].lower():
            return True
        for category in attribute['path']:
            if filterText in category.lower():
                return True

        return False

    def updateAttributes(self):
        """Use stored attributes to update widgets for current camera.

        If no stored attributes exist for the currently selected camera, then
            load them from the camera

        Returns:
            None

        """

        camSerial = self.getCurrentCamSerial()
        if camSerial is not None:
            attributeNames = sorted([attribute['displayName'] for attribute in self.storedAttributes[camSerial] if self.checkFilter(attribute)])
            self.attributeList['values'] = attributeNames
            if len(attributeNames) > 0:
                self.attributeVar.set(attributeNames[0])
            else:
                self.attributeVar.set('')

    def handleAttributeChange(self, *args, **kwargs):
        """React to a change in the attribute selection.

        Update value and GUI state.

        Args:
            *args: Unused event arguments.
            **kwargs: Unused event arguments.

        Returns:
            None

        """
        self.updateCategoryList()
        self.updateValue()
        self.updateApplyButtonState()

    def getValue(self):
        attribute = self.getCurrentAttribute()
        value = self.getValue()
        return value

    def updateValue(self):
        """Update value display to reflect the current stored value

        Also switch from entry to combobox depending on whether the selected
            attribute is an enum or not

        Returns:
            type: Description of returned object.

        """

        attribute = self.getCurrentAttribute()
        if attribute is None:
            self.valueVar.set('')
            return
        print('type=', attribute['type'])
        if attribute['type'] == 'enum':
            self.valueList['values'] = list(attribute['options'].values())
            self.valueEntry.grid_remove()
            self.valueList.grid()
        elif attribute['type'] == 'boolean':
            self.valueList['values'] = [str(True), str(False)]
            self.valueEntry.grid_remove()
            self.valueList.grid()
        else:
            self.valueList.grid_remove()
            self.valueEntry.grid()

        value = self.convertAttributeValue(attribute['value'], attribute['type'])

        if attribute['type'] == 'enum':
            self.valueVar.set(value)
        elif attribute['type'] == 'boolean':
            self.valueVar.set(str(value))
        else:
            self.valueVar.set(str(value))

    def getCurrentAttribute(self, displayName=None, modified=False):
        """Get an attribute from the list of stored attributes by displayName.

        If displayName is not given, the currently selected one is returned

        Args:
            displayName (str): displayName for selecting an attribute. Defaults to None.
            modified (bool): boolean flag indicating that the user's currently
                entered values should be used in place of stored ones

        Returns:
            dict: a dict representing a camera attribute

        """

        camSerial = self.getCurrentCamSerial()
        if camSerial is None:
            return None
        if displayName is None:
            displayName = self.attributeVar.get()
        for attribute in self.storedAttributes[camSerial]:
            if attribute['displayName'] == displayName:
                if modified:
                    if attribute['type'] == 'enum':
                        attribute['value'] = (attribute['value'][0], self.valueVar.get())
                    else:
                        attribute['value'] = self.valueVar.get()
                return attribute
        return None

    def updateApplyButtonState(self):
        """Disable/enable the "apply" button according to attribute read-only state.

        Returns:
            None

        """

        attribute = self.getCurrentAttribute()
        if attribute is None or attribute['accessMode'] == 'RO':
            # Attribute is read only - disable apply button
            self.applyConfigurationButton['state'] = tk.DISABLED
            self.valueEntry['state'] = tk.DISABLED
            self.addButton['state'] = tk.DISABLED
        elif attribute['accessMode'] == 'RW':
            # Attribute is read/write - enable apply button
            self.applyConfigurationButton['state'] = tk.NORMAL
            self.valueEntry['state'] = tk.NORMAL
            self.addButton['state'] = tk.NORMAL

    def addCurrentAttributeToConfiguration(self):
        attribute = self.getCurrentAttribute(modified=True)
        camSerial = self.getCurrentCamSerial()
        if camSerial is not None and attribute is not None:
            if attribute['type'] == 'enum':
                attribute = {'name':attribute['name'], 'value':attribute['value'][1], 'type':attribute['type']}
            else:
                attribute = {'name':attribute['name'], 'value':attribute['value'], 'type':attribute['type']}
            self.updateCurrentConfiguration({camSerial:{attribute["name"]:attribute}})

    def updateCurrentConfiguration(self, configuration):
        """Add one or more attributes to the configuration text.

        Args:
            configuration (dict of lists of dicts): A data structure containing
                one or more attributes organized like so:

                {camSerial1: {name1:attribute1, name2:attribute2, ...},
                 camSerial2: {name3:attribute3, name4:attribute4, ...},
                 ...}

            each attribute should be a dict containing at least the fields
            "value" and "type".

        Returns:
            None

        """
        currentConfiguration = self.getCurrentConfiguration()
        currentConfiguration = self.updateConfiguration(currentConfiguration, configuration)
        self.setCurrentConfiguration(currentConfiguration)

    def updateConfiguration(self, oldConfiguration, newConfiguration):
        for camSerial in newConfiguration:
            if camSerial not in oldConfiguration:
                oldConfiguration[camSerial] = {}
            for name in newConfiguration[camSerial]:
                oldConfiguration[camSerial][name] = newConfiguration[camSerial][name]
        return oldConfiguration

    def setCurrentConfiguration(self, configuration):
        configurationText = ''
        for camSerial in configuration:
            for name in configuration[camSerial]:
                configurationText += '{camSerial}, {name}, {value}, {type}\n'.format(camSerial=camSerial, name=name, value=configuration[camSerial][name]['value'], type=configuration[camSerial][name]['type'])
        self.configurationText.delete("1.0", tk.END)
        self.configurationText.insert("1.0", configurationText)
        self.configurationChangeHandler()

    def getCurrentConfiguration(self):
        configurationText = self.configurationText.get("1.0", tk.END).strip()
        attributes = configurationText.split('\n')
        configuration = {}
        for attribute in attributes:
            if len(attribute.strip()) == 0:
                continue
            elements = [element.strip() for element in attribute.split(',')]
            camSerial, name, value, type = elements
            if camSerial not in configuration:
                configuration[camSerial] = {}
            configuration[camSerial][name] = dict(name=name, value=value, type=type)
        return configuration

    def applyCurrentAttribute(self):
        attribute = self.getCurrentAttribute()
        camSerial = self.getCurrentCamSerial()

        if attribute is not None and camSerial is not None:
            name = attribute['name']
            type = attribute['type']
            value = attribute['value']
            value = self.convertAttributeValue(value, type)
            result = psu.setCameraAttribute(name, value, camSerial=camSerial, attributeType=type, nodemap='NodeMap')
            print('Applied attribute:', camSerial, name, value, type)

    def convertAttributeValue(self, value, type):
        if type == 'enum':
            value = value[1]
        elif type == 'integer':
            value = int(value)
        elif type == 'float':
            value = float(value)
        elif type == 'boolean':
            value = (value == 'True') or value == '1'
        return value

    def applyConfiguration(self):
        """Attempt to apply all configurations to the cameras now.

        Also reload camera settings afterward.

        Returns:
            None

        """

        configuration = self.getCurrentConfiguration()

        # Reformat configuration to {camSerial1:[(name1, value1, type1), (name2, value2, type2), ..., (nameN, valueN, typeN)], camSerial2:...}
        formattedConfiguration = {}
        for camSerial in configuration:
            formattedConfiguration[camSerial] = []
            for name in configuration[camSerial]:
                value = configuration[camSerial][name]['value']
                type =  configuration[camSerial][name]['type']
                value = self.convertAttributeValue(value)
                formattedConfiguration[camSerial].append(
                    (name, value, type)
                )

        for camSerial in formattedConfiguration:
            results = psu.setCameraAttributes(formattedConfiguration[camSerial], camSerial=camSerial)
            print('set attribute results for cam {cs}:'.format(cs=camSerial))
            print(results)

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

        attribute = self.getCurrentAttribute()
        if attribute is None:
            self.categoryVar.set('')
        else:
            path = ' > '.join(attribute['path'][1:])
            self.categoryVar.set(path)
