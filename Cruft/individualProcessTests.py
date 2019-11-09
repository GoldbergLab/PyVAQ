import PyVAQ as p
import multiprocessing as mp
import time
import queue

r'''
cd "C:\Users\Brian Kardon\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Video VI\PyVAQ"
python individualProcessTests.py
'''

if __name__ == '__main__':
    print("Begin individual test")
    aq = mp.Queue()
    aMonq = mp.Queue()
    aaMesq = mp.Queue()
    awMesq = mp.Queue()
    ready = mp.Barrier(4)
    sMesq = mp.Queue()
    audioDAQChannels = ['Dev3/ai5']
    camSerials = p.discoverCameras()
    print(camSerials)
    vq = mp.Queue()
    videoFrameRate = 60
    audioFrequency = 44100
    vaMesq = mp.Queue()
    vwMesq = mp.Queue()
    mMesq = mp.Queue()
    bufferSizeAudioChunks = 2
    chunkSize = 44100
    bufferSizeSeconds = bufferSizeAudioChunks * chunkSize / audioFrequency
    exposureTime = 10000

    if exposureTime >= 0.95*1/videoFrameRate:
        print()
        print("******WARNING*******")
        print()
        print("Exposure time is too long to achieve requested frame rate!")
        print("Setting exposure time lower.")
        print()
        print("********************")
        print()
        exposureTime = 1000000*0.95/videoFrameRate

    acquireSettings = [
                        ('AcquisitionMode', 'Continuous', 'enum'),
                        ('TriggerMode', 'Off', 'enum'),
                        ('TriggerSelector', 'FrameStart', 'enum'),
                        ('TriggerSource', 'Line0', 'enum'),
                        ('TriggerActivation', 'RisingEdge', 'enum'),
                        # ('ExposureMode', 'TriggerWidth'),
                        # ('Width', 800, 'integer'),
                        # ('Height', 800, 'integer'),
                        ('TriggerMode', 'On', 'enum'),
                        ('ExposureAuto', 'Off', 'enum'),
                        ('ExposureMode', 'Timed', 'enum'),
                        ('ExposureTime', exposureTime, 'float')]   # List of attribute/value pairs to be applied to the camera in the given order

    s = p.Synchronizer(audioSyncChannel='Dev3/ctr0',
                    videoSyncChannel='Dev3/ctr1',
                    audioFrequency=audioFrequency,
                    videoFrequency=videoFrameRate,
                    messageQueue=sMesq,
                    verbose=False,
                    ready=ready)
    aa = p.AudioAcquirer(
        audioQueue=aq,
        audioMonitorQueue=aMonq,
        messageQueue=aaMesq,
        chunkSize=chunkSize,
        samplingRate=audioFrequency,
        bufferSize=None,
        channelNames=audioDAQChannels,
        syncChannel='PFI4',
        verbose=False,
        ready=ready)
    aw = p.AudioWriter(
        wavBaseFilename='indiv_test',
        audioQueue=aq,
        messageQueue=awMesq,
        mergeQueue=mMesq,
        chunkSize=chunkSize,
        bufferSizeSeconds=bufferSizeSeconds,
        audioFrequency=44100,
        numChannels=len(audioDAQChannels),
        verbose=False)
    va = p.VideoAcquirer(
        camSerial=camSerials[0],
        imageQueue=vq,
        monitorImageQueue=None,
        acquireSettings=acquireSettings,
        frameRate=videoFrameRate,
        monitorFrameRate=1,
        messageQueue=vaMesq,
        verbose=False,
        stdoutLock=None,
        ready=ready)

    vw = p.VideoWriter(
        aviBaseFilename='vtest',
        imageQueue=vq,
        frameRate=videoFrameRate,
        messageQueue=vwMesq,
        mergeQueue=mMesq,
        bufferSizeSeconds=bufferSizeSeconds,
        verbose=2,
        stdoutLock=None
        )

    m = p.AVMerger(
        messageQueue=mMesq,
        verbose=True,
        stdoutLock=None
        )

    print("Start audioWriter")
    aw.start()
    awMesq.put((p.AudioWriter.START, None))

    print("Start audioAcquirer")
    aa.start()
    aaMesq.put((p.AudioAcquirer.START, None))

    print("Start VideoWriter")
    vw.start()
    vwMesq.put((p.VideoWriter.START, None))

    print("Start videoAcquirer")
    va.start()
    vaMesq.put((p.VideoAcquirer.START, None))

    print("Start sync")
    s.start()
    sMesq.put((p.Synchronizer.START, None))
    ready.wait()

    print("Start merger")
    m.start()
    mMesq.put((p.AVMerger.START, None))

    time.sleep(2)

    while True:
        x = input(">>> ")
        if len(x) > 0:
            break
        # print("Send trigger")
        t = time.time_ns()/1000000000
        trig = p.Trigger(t-2, t, t+2)
        vwMesq.put((p.VideoWriter.TRIGGER, trig))
        awMesq.put((p.AudioWriter.TRIGGER, trig))

    print("Stopping everything")
    sMesq.put((p.Synchronizer.STOP, None))
    sMesq.put((p.Synchronizer.EXIT, None))
    # aaMesq.put((p.AudioAcquirer.STOP, None))
    # aaMesq.put((p.AudioAcquirer.EXIT, None))
    # awMesq.put((p.AudioWriter.STOP, None))
    # awMesq.put((p.AudioWriter.EXIT, None))
    vaMesq.put((p.VideoAcquirer.STOP, None))
    vaMesq.put((p.VideoAcquirer.EXIT, None))

    time.sleep(2)
