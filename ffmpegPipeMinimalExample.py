import numpy as np
import subprocess

# Initialize variables
numFrames = 50
w = 1200
h = 1080
filename = 'test.avi'
shapeArg = '{w}x{h}'.format(w=w, h=h)

# Open ffmpeg process
ffmpegProc = subprocess.Popen(
    ['ffmpeg', '-hide_banner', '-y', '-v', 'error',
     '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', shapeArg,
     '-r', '30', '-i', 'pipe:', '-c:v', 'libx264',
     '-an', filename], stdin=subprocess.PIPE, bufsize=0)

while True:
    # Send a bunch of frames to ffmpeg to write to disk
    for f in range(numFrames):
        # Generate a blank image to send to ffmpeg
        dummyFrame = np.random.randint(0, 255, [w, h, 3], dtype='uint8')
        ffmpegProc.stdin.write(dummyFrame.tobytes())
#    ffmpegProc.stdin.flush()
    # At this point, file size = 6 kB - seems like a lot for header info, but not enough for video?

    x = input('Ready? ')
    if x == "y":
        break;

# Wait for ffmpeg to terminate
print('communicating...')
ffmpegProc.stdin.close()
#ffmpegProc.communicate()
print('...communicated')
# At this point, file size = 66 kB
