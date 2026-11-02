# PyVAQ

Synchronized multiple video/audio acquisition software, written in python. A GUI allows acquisition and file writing control and monitoring. Video acquisition and writing are done asynchronously to avoid missing data. An arbitrarily-sized pre-buffer can be kept (limited by system memory resources). Triggering can be done manually (with a button in the GUI), continuously, or automatically with a volume-based trigger.

## Multiprocessing

Video/Audio acquisition and writing are done in separate state-machine processes, taking advantage of multi-core architectures, if present.

## Synchronization

Synchronization is done using the same National Instruments data acquisition device (DAQ) as is used for audio capture. The DAQ generates two clock signals, one at the requested audio frequency, and one at the requested video frequency. The audio frequency clock signal is used internally by the DAQ to trigger audio acquisition. The video frequency clock must be routed from one of the DAQ's counter outputs to the camera's hardware trigger input, and the camera must be configured to use an external hardware trigger for each frame grab.

## Audio capture

Audio capture is designed for National Instruments USB series of data acquisition devices. This code is tested on model USB-6002 devices, but should theoretically work for any USB DAQs, and possibly other series as well, as long as the DAQ has at least two "counter" outputs, and as many differential analog inputs as desired audio channels.

## Video capture

Video capture is designed for [FLIR Blackfly S USB cameras](https://www.flir.com/products/blackfly-s-usb3/?vertical=machine+vision&segment=iis), but should theoretically work with any USB3-vision-conforming camera with a hardware trigger input. Experimental support also exists for generic DirectShow/OpenCV cameras (webcams), Aptina/onsemi sensor boards, ams-OSRAM NanEye cameras, and Basler cameras (see "Optional camera backends" below).

## AV merging

The audio streams are written to disk as a single multi-track .wav file, and each video stream is written to disk separately. On-line asynchronous merging of audio and video can be configured in the GUI.

## Requirements and Dependencies

 - Windows 10
 - [Python 3.10](https://www.python.org/downloads/release/python-31011/) (64-bit). Python 3.10 is required because the Spinnaker python wheel is built for a specific Python version.
   - Various python libraries. See requirements.txt for the pinned non-standard-library python dependencies available on PyPI.\*
   - [spinnaker-python](https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/iis/spinnaker-sdk-download/) v4.3.0.190 for CPython 3.10, x64 Windows (python wrapper for the Spinnaker SDK)\*\*
 - [Spinnaker SDK](https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/iis/spinnaker-sdk-download/) v4.3.0.190 (FLIR camera drivers & SDK)
 - ffmpeg\*\*\*, installed and on system path
 - [NI-DAQmx](https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html) - drivers for NI DAQs
 - NI DAQ, such as [USB-6002](https://www.ni.com/docs/en-US/bundle/usb-6002-specs/resource/374371a.pdf)
 - Microphone and amplifier, connected to analog input on DAQ
 - One or more [FLIR Blackfly S USB camera](https://www.flir.com/products/blackfly-s-usb3/?vertical=machine+vision&segment=iis) or other FLIR camera with a similar interface
   - Cable to connect camera GPIO port to DAQ for hardware triggering

\* = Use `pip install -r requirements.txt` to install. Note that numpy is pinned below 2.0 because the Spinnaker 4.3 python wheel is compiled against the NumPy 1.x ABI; upgrading numpy will break `import PySpin`.

\*\* = Not available on PyPI, must be downloaded from Teledyne FLIR and installed with pip from the .whl file.

\*\*\* = If GPU-accelerated video encoding is desired, an [NVIDIA GPU with one or more NVENC cores](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new) must be present, and the [ffmpeg build must include the h264_nvenc codec](https://www.gyan.dev/ffmpeg/builds/).

### Optional camera backends

These are only needed if you want to use the corresponding camera type. Everything in requirements.txt is installed regardless; the items below are not.

 - **OpenCV / DirectShow cameras (webcams):** no extra install. `opencv-python` and `pygrabber` are in requirements.txt.
 - **Basler cameras:** install the [pylon Camera Software Suite](https://www.baslerweb.com/en/software/pylon/) and `pip install pypylon`.
 - **ams-OSRAM NanEye cameras:** copy the Awaiba/NanEye .NET SDK DLLs (`Awaiba.dll` and its dependencies) into `lib\NanEye` inside the PyVAQ folder. They are loaded via `pythonnet`, which is in requirements.txt.

## Known issues

Recording scheduling function does not yet work.

## Installation

1. Install GitHub Desktop (or git)
2. Clone [PyVAQ repository to your computer](https://github.com/GoldbergLab/PyVAQ)
3. Install NI-DAQmx
4. Install Python 3.10 (64-bit)
5. Create a virtual environment and install python libraries:
	a) Open command prompt
	b) cd C:\path\to\where\PyVAQ\is
	c) py -3.10 -m venv .venv
	d) .venv\Scripts\activate
	e) pip install -r requirements.txt
6. Install Spinnaker SDK (version 4.3.0.190 for x64 Windows)
7. Install Spinnaker python library (version 4.3.0.190 for CPython 3.10, x64 Windows), with the virtual environment still activated:
	a) Unzip spinnaker-python
	b) pip install C:\path\to\unzipped\spinnaker_python-4.3.0.190-cp310-cp310-win_amd64.whl
8. Install ffmpeg with GPU support
	a) Unzip ffmpeg
	b) Move to C:\Program Files
	c) Add ffmpeg path to system Path (environment variable)
9. (Optional) Install any extra camera backends listed under "Optional camera backends" above.
10. Run PyVAQ with the virtual environment activated: `python PyVAQ.py`

## Author

Developed by Brian Kardon (edu.llenroc@72kmb or moc.liamg@nodrak.nairb) 2019
