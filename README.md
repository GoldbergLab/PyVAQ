# PyVAQ

Synchronized multiple video/audio acquisition software, lovingly written in python, designed for National Instruments USB DAQs and FLIR Blackfly S cameras (but will probably work with any USB3-vision-conforming camera).

## Multiprocessing

Video/Audio acquisition and writing are done in separate state-machine processes, taking advantage of multi-core architectures, if present.
