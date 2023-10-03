# PyVAQ

Synchronized multiple video/audio acquisition software, written in python. A GUI allows acquisition and file writing control and monitoring. Video acquisition and writing are done asynchronously to avoid missing data. An arbitrarily-sized pre-buffer can be kept (limited by system memory resources). Triggering can be done manually (with a button in the GUI), continuously, or automatically with a volume-based trigger.

## Multiprocessing

Video/Audio acquisition and writing are done in separate state-machine processes, taking advantage of multi-core architectures, if present.

## Synchronization

Synchronization is done using the same National Instruments data acquisition device (DAQ) as is used for audio capture. The DAQ generates two clock signals, one at the requested audio frequency, and one at the requested video frequency. The audio frequency clock signal is used internally by the DAQ to trigger audio acquisition. The video frequency clock must be routed from one of the DAQ's counter outputs to the camera's hardware trigger input, and the camera must be configured to use an external hardware trigger for each frame grab.

## Audio capture

Audio capture is designed for National Instruments USB series of data acquisition devices. This code is tested on model USB-6002 devices, but should theoretically work for any USB DAQs, and possibly other series as well, as long as the DAQ has at least two "counter" outputs, and as many differential analog inputs as desired audio channels.

## Video capture

Video capture is designed for [FLIR Blackfly S USB cameras](https://www.flir.com/products/blackfly-s-usb3/?vertical=machine+vision&segment=iis), but should theoretically work with any USB3-vision-conforming camera with a hardware trigger input.

## AV merging

The audio streams are written to disk as a single multi-track .wav file, and each video stream is written to disk separately. On-line asynchronous merging of audio and video can be configured in the GUI.

## Requirements and Dependencies

 - Windows 10
 - [Python 3.8.10](https://www.python.org/downloads/release/python-3810/)
   - Various python libraries. See requirements.txt file for non-standard-library python dependencies available on common online python library repositories.\*
   - [spinnaker-python](https://www.flir.com/support-center/iis/machine-vision/downloads/spinnaker-sdk-and-firmware-download/) library v2.4.0.144 for python 3.8.10 (python wrapper for Spinnaker SDK)\*\*
 - [Spinnaker SDK](https://www.flir.com/support-center/iis/machine-vision/downloads/spinnaker-sdk-and-firmware-download/) v2.4.0.144 (FLIR camera drivers & SDK)
 - [ffmpeg](https://ffmpeg.org/)\*\*\*, installed and on system path
 - [NI DAQmx v18.6](https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html#291872) - drivers for NI DAQs
 - NI DAQ, such as [USB-6002](https://www.ni.com/docs/en-US/bundle/usb-6002-specs/resource/374371a.pdf)
 - Microphone and amplifier, connected to analog input on DAQ
 - One or more [FLIR Blackfly S USB camera](https://www.flir.com/products/blackfly-s-usb3/?vertical=machine+vision&segment=iis) or other FLIR camera with a similar interface
   - [Cable to connect camera GPIO port to DAQ for hardware triggering](https://www.flir.com/products/hirose-hr10-6-pin-circular-connector/)

\* = Use pip install -r requirements.txt to install

\*\* = Not available in online repositories, must be downloaded from FLIR and installed with pip from .whl file.

\*\*\* = If GPU-accelerated video encoding is desired, an [NVIDIA GPU with one or more NVENC cores](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new) must be present, and the [ffmpeg build must include the h264_nvenc codec](https://www.gyan.dev/ffmpeg/builds/).
## Known issues

Recording scheduling function does not yet work.

## Installation

1. Install GitHub Desktop (or git)
2. Clone [PyVAQ repository to your computer](https://github.com/GoldbergLab/PyVAQ)
3. Install NI-DAQmx
4. Install Python 3.8.10
5. Install python libraries:

    a) Open command prompt
   
    b) cd C:\path\to\where\PyVAQ\is

    c) pip install -r requirements.txt

7. Install Spinnaker SDK (version 2.4.0.144 for x64 Windows)

    a) Select the "Application Development" installation profile when prompted

8. Install Spinnaker python library (version 2.4.0.144 for CPython 3.8.10, x64 Windows):

    a) Unzip spinnaker-python

    b) Open command prompt

    c) cd C:\path\to\where\unzipped\spinnaker-python\is

    d) pip install spinnaker_python-2.4.0.144-cp38-cp38-win_amd64.whl

9. Install ffmpeg with GPU support

    a) Unzip ffmpeg

    b) Move to C:\ProgramFiles

    c) Add ffmpeg path to system Path (environment variable)

## Author

Developed by Brian Kardon (edu.llenroc@72kmb or moc.liamg@nodrak.nairb) 2019
