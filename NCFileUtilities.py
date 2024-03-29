from netCDF4 import Dataset
import numpy as np

def readNCFile(path):
    """Read a netCDF file formatted for electro_gui.

    Args:
        path (str): Path to a .nc file.

    Returns:
        data (dict): A dictionary of fields and values from the .nc file.

    """
    # Open the .nc file
    dataset = Dataset(path, 'r')
    # Initialize the data dictionary
    data = {}
    # Prepare the list of fields to read
    fields = ['time', 'dt', 'chan', 'metaData', 'data']
    for field in fields:
        # Read the value for each field in the .nc file
        data[field] = dataset.variables[field][:]
    return data

class NCFile:
    """A class providing a persistent interface to an .nc file.

    This is intended to be used to consecutively add data onto an .nc file. A
        .nc file is a data file designed to be read by electro_gui.

    Args:
        path (str): Path to save file to.
        time (list): A 7 element list of integers representing a time vector, in
            the order [year, month, day, hour, minute, second, microsecond].
        dt (float): the data sampling period (time between consecutive samples)
        chan (int): the channel number for the data
        metaData (str): arbitrary descriptive string
        dataType (str): A data type specifier to use to define the data field
            in the .nc file. If left as None, and data is a numpy array, the
            data type of the array will be used. Otherwise the data type will
            default to float.

    Attributes:
        dataset (netCDF4.Dataset): A netCDF4.Dataset object.
        dataVar (netCDF4.Variable): A netCDF4.Variable object containing the
            data for the .nc file.
        idx (int): The current data insertion index
        path
        time
        dt
        chan
        metaData
        dataType

    """

    def __init__(self, path, time, dt, chan, metaData, dataType=None):
        self.path = path
        self.time = time
        self.dt = dt
        self.chan = chan
        self.metaData = metaData
        self.dataType = dataType

        # Maximum length of metaData character string
        self.metaDataLength = 256

        # Initialize file
        self.dataset = Dataset(self.path, mode='w', clobber=True, format='NETCDF3_CLASSIC')

        self.createVariables()

        self.fillVariables()

        # Keep track of how much data we've written
        self.idx = 0

    def fillVariables(self):
        # Add data
        self.timeVectorVar[:] = self.time
        self.deltaTVar[:] = self.dt
        self.channelVar[:] = self.chan
        self.metaDataVar[:len(self.metaData)] = self.metaData

    def createVariables(self):
        # Define variables that will go into file
        self.createTimeVectorVar()
        self.createDeltaTVar()
        self.createChannelVar()
        self.createMetaDataVar()
        self.dataVar = None
    def createTimeVectorVar(self):
        timeVectorDim = self.dataset.createDimension('ti', size=7)
        self.timeVectorVar = self.dataset.createVariable('time', 'i', dimensions=(timeVectorDim,))   # i = NC_INT
    def createDeltaTVar(self):
        self.deltaTVar =     self.dataset.createVariable('dt', 'd', dimensions=())                   # d = NC_DOUBLE
    def createChannelVar(self):
        self.channelVar = self.dataset.createVariable('chan', 'i', dimensions=())
    def createMetaDataVar(self):
        metaDataDim =   self.dataset.createDimension('md', size=self.metaDataLength)
        self.metaDataVar =   self.dataset.createVariable('metaData', 'c', dimensions=(metaDataDim,)) # c = NC_CHAR

    def initializeDataVar(self, sampleData):
        # Set up dataVar the first time data is added

        if self.dataType is None:
            # Attempt to get data type from data as if it were a numpy array.
            #   If it is not, then default to 'f' (float)
            try:
                self.dataType = str(data.dtype)
            except AttributeError:
                self.dataType = 'f'

        unlimDim =      self.dataset.createDimension('da', size=None)
        self.dataVar =  self.dataset.createVariable('data', self.dataType, dimensions=(unlimDim, ))

    def addData(self, data):
        if not self.dataset.isopen():
            raise IOError('Dataset is already closed, cannot add data.')

        if self.dataVar is None:
            self.initializeDataVar(data)

        # Append data
        self.dataVar[self.idx:self.idx+len(data), ...] = data

        # Update current data insertion index
        self.idx += len(data)

    def close(self):
        # Close file
        self.dataset.close()

class NCFileMultiChannel(NCFile):
    def __init__(self, path, time, dt, channels, metaData, dataType=None):
        super().__init__(path, time, dt, channels, metaData, dataType=dataType)
    def createChannelVar(self):
        self.channelDim = self.dataset.createDimension('ch', size=len(self.chan))
        self.channelVar = self.dataset.createVariable('chan', 'i', dimensions=(self.channelDim,))
    def initializeDataVar(self, sampleData):
        # Set up dataVar the first time data is added

        if self.dataType is None:
            # Attempt to get data type from data as if it were a numpy array.
            #   If it is not, then default to 'f' (float)
            try:
                self.dataType = str(sampleData.dtype)
            except AttributeError:
                self.dataType = 'f'

        unlimDim =      self.dataset.createDimension('da', size=None)
        self.dataVar =  self.dataset.createVariable('data', self.dataType, dimensions=(unlimDim, self.channelDim))

def writeNCFile(path, time, dt, chan, metaData, data, dataType=None):
    """Write a netCDF file formatted so electro_gui can open it.

    Args:
        path (str): Path to save file to.
        time (list): A 7 element list of integers representing a time vector, in
            the order [year, month, day, hour, minute, second, microsecond].
        dt (float): the data sampling period (time between consecutive samples)
        chan (int): the channel number for the data
        metaData (str): arbitrary descriptive string
        data (iterable): an iterable containing the data to be written,
            typically either a numpy array or a list
        dataType (str, numpy.dtype, ...): A data type specifier to use to define
            the data field in the .nc file. If left as None, and data is a
            numpy array, the dtype of the array will be used. Otherwise the
            dtype will default to float.

    Returns:
        None

    """
    # Initialize file
    dataset = Dataset(path, mode='w', clobber=True, format='NETCDF3_CLASSIC')

    # Maximum length of metaData character string
    metaDataLength = 64

    if dataType is None:
        # Attempt to get data type from data as if it were a numpy array.
        #   If it is not, then default to 'f' (float)
        try:
            dataType = str(data.dtype)
        except AttributeError:
            dataType = 'f'

    # Define variables that will go into file
    timeVectorDim = dataset.createDimension('ti', size=7)
    timeVectorVar = dataset.createVariable('time', 'i', dimensions=(timeVectorDim,))   # i = NC_INT
    deltaTVar =     dataset.createVariable('dt', 'd', dimensions=())                   # d = NC_DOUBLE
    channelVar =    dataset.createVariable('chan', 'i', dimensions=())
    metaDataDim =   dataset.createDimension('md', size=metaDataLength)
    metaDataVar =   dataset.createVariable('metaData', 'c', dimensions=(metaDataDim,)) # c = NC_CHAR
    unlimDim =      dataset.createDimension('da', size=None)
    dataVar =       dataset.createVariable('data', dataType, dimensions=(unlimDim, ))

    # Add data
    timeVectorVar[:] = time
    deltaTVar[:] = dt
    channelVar[:] = chan
    metaDataVar[:len(metaData)] = metaData
    dataVar[:] = data

    # Close file
    dataset.close()

def extractBooleanDataFromDigitalArray(digitalArray, lineNumbers):
    """Extract boolean data from a digital array of integers.

    NI DAQmx returns digital reads as integers, where the nth bit of the integer
        corresponds to the read value of the nth line in the port. This function
        converts an integer array of reads and extracts a simple boolean array
        where the (m, n) element is the nth line value on the mth sample

    Args:
        digitalArray (1D array of int): An integer array of digital values
        lineNumbers (1D array of int): An integer array of line numbers to
            extract

    Returns:
        2D boolean array: The extracted digital values.

    """

    return (np.right_shift.outer(digitalArray, lineNumbers) & 1).astype('bool')

if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    print('NCFile test:')
    path = Path.home() / r'Downloads\test.nc'
    time = [1, 2, 3, 4, 5, 6, 7]
    dt = 0.123
    channels = 0 #[0, 1, 2, 3]
    metaData = 'testing 1 2 3'
    data = np.random.randint(100, size=441000, dtype='uint32')
    writeNCFile(path, time, dt, channels, metaData, data, dataType='i4')
    # f = NCFileMultiChannel(path, time, dt, channels, metaData)
    # data = np.random.randint(100, size=(20, len(channels)), dtype='uint8')
    # f.addData(data)
    # f.close()
    print('done')
