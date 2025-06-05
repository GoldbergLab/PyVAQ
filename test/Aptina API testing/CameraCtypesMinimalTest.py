import ctypes

# Load the DLL. Adjust the path as needed.
dll_apbase = ctypes.WinDLL(r"C:\Aptina Imaging\bin\apbase.dll")

# Define custom ctypes types
ap_s32 = ctypes.c_int

dll_apbase.ap_DeviceProbe.argtypes = [ctypes.c_char_p]
dll_apbase.ap_DeviceProbe.restype = ap_s32

print('before')
print(dll_apbase.__dict__.keys())

device_dir = r'C:\Aptina Imaging\sensor_data\MT9V024-REV4.xsdat'
device_dir_p = ctypes.c_char_p(device_dir.encode('utf-8'))
err = dll_apbase.ap_DeviceProbe(device_dir_p)

print('after')
print(dll_apbase.__dict__.keys())
