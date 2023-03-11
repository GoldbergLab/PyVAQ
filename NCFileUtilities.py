from netCDF4 import Dataset

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

def writeNCFile(path, time, dt, chan, metaData, data):
    """Write a netCDF file formatted so electro_gui can open it.

    Args:
        path (str): Path to save file to.
        time (list): A 7 element list of integers representing a time vector, in
            the order [year, month, day, hour, minute, second, microsecond].
        dt (float): the data sampling period (time between consecutive samples)
        chan (int): the channel number for the data
        metaData (str): arbitrary descriptive string
        data (iterable): an iterable containing the data to be written

    Returns:
        None

    """
    # Initialize file
    dataset = Dataset(path, mode='w', clobber=True, format='NETCDF3_CLASSIC')

    # Maximum length of metaData character string
    metaDataLength = 64;

    # Define variables that will go into file
    timeVectorDim = dataset.createDimension('p', size=7);
    timeVectorVar = dataset.createVariable('time', 'i', dimensions=(timeVectorDim,));   # i = NC_INT
    deltaTVar =     dataset.createVariable('dt', 'd', dimensions=());                   # d = NC_DOUBLE
    channelVar =    dataset.createVariable('chan', 'i', dimensions=());
    metaDataDim =   dataset.createDimension('c', size=metaDataLength);
    metaDataVar =   dataset.createVariable('metaData', 'c', dimensions=(metaDataDim,)); # c = NC_CHAR
    unlimDim =      dataset.createDimension('t', size=None);
    dataVar =       dataset.createVariable('data', 'f', dimensions=(unlimDim, )); # f = NF_FLOAT

    # Add data
    timeVectorVar[:] = time
    deltaTVar[:] = dt
    channelVar[:] = chan
    metaDataVar[:len(metaData)] = metaData
    dataVar[:] = data

    # Close file
    dataset.close();
