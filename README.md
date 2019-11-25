# PyVAQ

Synchronized multiple video/audio acquisition software, lovingly written in python. A GUI allows acquisition and writing control and parameter settings, as well as monitoring the audio and video streams. Video acquisition and writing are done asynchronously to avoid missing data. An arbitrarily-sized pre-buffer can be kept (limited by system memory resources). Triggering can be done manually (with a button in the GUI) or automatically with a volume-based trigger.

## Multiprocessing

Video/Audio acquisition and writing are done in separate state-machine processes, taking advantage of multi-core architectures, if present.

## Audio capture

Audio capture is designed for National Instruments USB series of data acquisition devices (DAQs). This code is tested on model USB-6002 devices, but should theoretically work for any USB DAQs, and possibly other series as well, as long as the DAQ has at least two "counter" outputs, and as many differential analog inputs as desired audio channels.

## Video capture

Video capture is designed for FLIR Blackfly S USB cameras, but should theoretically work with any USB3-vision-conforming camera.

## AV merging

The audio streams are written to disk as a single multi-track .wav file, and each video stream is written to disk separately. On-line asynchronous merging of audio and video can be configured in the GUI.

By Brian Kardon (edu.llenroc@72kmb or moc.liamg@nodrak.nairb)
