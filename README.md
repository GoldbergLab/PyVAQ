# PyVAQ

Synchronized multiple video/audio acquisition software, lovingly written in python. A GUI allows acquisition and writing control and parameter settings, as well as monitoring the audio and video streams. Video acquisition and writing are done asynchronously to avoid missing data. An arbitrarily-sized pre-buffer can be kept (limited by system memory resources). Triggering can be done manually (with a button in the GUI) or automatically with a volume-based trigger.

## Multiprocessing

Video/Audio acquisition and writing are done in separate state-machine processes, taking advantage of multi-core architectures, if present.

## Synchronization

Synchronization is done using the same National Instruments data acquisition device (DAQ) as is used for audio capture. The DAQ generates two clock signals, one at the requested audio frequency, and one at the requested video frequency. The audio frequency clock signal is used internally by the DAQ to trigger audio acquisition. The video frequency clock must be routed from one of the DAQ's counter outputs to the camera's hardware trigger input, and the camera must be configured to use an external hardware trigger for each frame grab.

## Audio capture

Audio capture is designed for National Instruments USB series of data acquisition devices. This code is tested on model USB-6002 devices, but should theoretically work for any USB DAQs, and possibly other series as well, as long as the DAQ has at least two "counter" outputs, and as many differential analog inputs as desired audio channels.

## Video capture

Video capture is designed for FLIR Blackfly S USB cameras, but should theoretically work with any USB3-vision-conforming camera with a hardware trigger input.

## AV merging

The audio streams are written to disk as a single multi-track .wav file, and each video stream is written to disk separately. On-line asynchronous merging of audio and video can be configured in the GUI.

## Non-standard library dependencies

numpy (image buffer handling, audio processing)
scipy (audio filtering)
Pillow (PIL) (GUI image display)
matplotlib (GUI display of audio stream and statistics)
pympler (finding memory leaks - will be removed when all the bugs are gone)
nidaqmx (interacting with the NI DAQ - acquiring audio and synchronizing audio/video)
PySpin (interacting with the FLIR Blackfly S USB camera)

## Known issues

 - Memory-related lock-up and sometimes crashes occur after several hours of operation due to a memory leak (still working on it!)


By Brian Kardon (edu.llenroc@72kmb or moc.liamg@nodrak.nairb) 2019
