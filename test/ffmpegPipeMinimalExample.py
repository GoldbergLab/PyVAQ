import numpy as np
import subprocess
import sys
import time
from scipy.ndimage import gaussian_filter, uniform_filter

# Initialize variables
numFrames = 100
w = 1200
h = 1080
filename = sys.argv[1]
shapeArg = '{w}x{h}'.format(w=w, h=h)

dummyFrames = np.random.randint(0, 255, [h, w, 3, numFrames], dtype='uint8')
for k in range(numFrames):
    a = 5*k
    b = 5*k + 200
    dummyFrames[a:b, a:b, 0, k] = 255
    dummyFrames[a:b, a:b, 1, k] = 255
    dummyFrames[a:b, a:b, 2, k] = 0
print('Created dummy frames')
dummyFrames = uniform_filter(dummyFrames, size=3)
print('Smoothed')

# Open ffmpeg process

FFMPEG_EXE = 'ffmpeg'

cpuCommand = [FFMPEG_EXE, '-hide_banner', '-y', '-v', 'verbose',
 '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', shapeArg,
 '-r', '30', '-i', 'pipe:', '-c:v', 'libx264',
 '-an', filename]

gpuCommand = [FFMPEG_EXE, '-y',
    '-vsync', 'passthrough', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
    '-v', 'verbose', '-f', 'rawvideo', '-c:v', 'rawvideo',
    '-pix_fmt', 'rgb24', '-s', shapeArg, '-thread_queue_size', '1024',
    '-r', '30', '-i', '-', '-c:v', 'h264_nvenc', '-preset', 'fast',
    '-cq', '40',
    '-pix_fmt', 'rgb0', '-an', filename]

ffmpegProc = subprocess.Popen(gpuCommand, stdin=subprocess.PIPE, bufsize=0)

startTime = time.time()
# Send a bunch of frames to ffmpeg to write to disk
for f in range(numFrames):
    # print('{f} of {n}'.format(f=f, n=numFrames))
    # Generate a blank image to send to ffmpeg
    dummyFrame = dummyFrames[:, :, :, f].tobytes()
    ffmpegProc.stdin.write(dummyFrame)
#    ffmpegProc.stdin.flush()
    # At this point, file size = 6 kB - seems like a lot for header info, but not enough for video?

# Wait for ffmpeg to terminate
ffmpegProc.stdin.close()
#ffmpegProc.communicate()
# At this point, file size = 66 kB
ffmpegProc.wait()
stopTime = time.time()


print('Elapsed time for {f} frames: {t}'.format(f=numFrames, t=stopTime-startTime))
print('FPS: {fps}'.format(fps=numFrames/(stopTime-startTime)))
