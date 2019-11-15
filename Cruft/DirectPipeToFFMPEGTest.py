# import subprocess
from PIL import Image
import numpy as np
# import struct
# cloudFilePath = r'C:\Users\Brian Kardon\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Video VI\PyVAQ\Source\Cruft\clouds.jpg'
# clouds = Image.open(cloudFilePath)
# w, h = clouds.size
# shapeArg = '{w}x{h}'.format(w=w, h=h)
# # print(shapeArg)
# # cloudsArray = b''.join([struct.pack('BBB', r, g, b) for r, g, b in clouds.getdata()])
# ffmpegProc = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', shapeArg, '-r', '20', '-i', 'pipe:', '-c:v', 'libx264', '-an', 'pipetestout.avi'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
#
# for k in range(360):
#     im = Image.blend(clouds.rotate(k), clouds.rotate(-k), 0.5)
#     ffmpegProc.stdin.write(im.tobytes('raw', 'RGB'))
# input("Is ffmpeg already going? ")
# ffmpegProc.communicate()

cloudFilePath = r'C:\Users\Brian Kardon\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Video VI\PyVAQ\Source\Cruft\clouds.jpg'
clouds = Image.open(cloudFilePath)

import ffmpegWriter as fw
f = fw.ffmpegWriter('ffmpegWriterTest.avi')
for k in range(360):
    im = Image.blend(clouds.rotate(k), clouds.rotate(-k), 0.5)
    # imarray = np.asarray(im)
    f.write(im)
f.close()
