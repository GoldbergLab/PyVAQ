import subprocess
import numpy as np
import time
import cv2
import traceback

DEBUG=False

def debug(*args, **kwargs):
    if DEBUG:
        print('*** ApSpin DEBUG ***')
        print(*args, **kwargs)
        print('*** ************ ***')
        print()

class ffplayer:
    def __init__(self, maxFrameRate, windowTitle='ffplayer', pixelFormat='rgb24'):
        self.width = None
        self.height = None
        self.maxFrameRate = maxFrameRate
        self.windowTitle = windowTitle
        self.ffplay_cmd = None
        self.ffplay_proc = None
        self.pixelFormat = pixelFormat

    def blank(self):
        # Display a blank frame indicating lack of image data
        if self.width is not None and self.height is not None:
            msg = 'no image data'
            blankFrame = np.zeros([self.height, self.width, 3], dtype='uint8')
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 255)
            thickness = 2
            (textWidth, textHeight), baseline = cv2.getTextSize(msg, font, fontScale, thickness)
            origin = ((self.width - textWidth)//2, (self.height + textHeight)//2)
            blankFrame = cv2.putText(blankFrame, msg, origin, font, fontScale, color, thickness)
            self.showFrame(blankFrame)

    def initialize(self, width, height):
        """Set up the ffplay process.

        Args:
            width (int): Width of expected video frames
            height (int): Height of expected video frames

        Returns:
            type: Description of returned object.

        """
        self.width = width
        self.height = height
        debug('Initializing ffplayer with width', self.width, 'height', self.height, 'pixelFormat', self.pixelFormat, 'maxFrameRate*2', self.maxFrameRate*2, 'windowTitle', self.windowTitle)

        self.ffplay_cmd = [
            'ffplay',
            '-f', 'rawvideo',
            '-pixel_format', self.pixelFormat,
            '-video_size', f'{self.width}x{self.height}',
            '-framerate', str(self.maxFrameRate*2),
            '-i', '-',
            # '-vf', f'drawtext=fontfile=\'C\\:/Windows/Fonts/Arial.ttf\':text={text}:fontcolor={color}:fontsize={fontsize}:x={textX}:y={textY}',  # ensure scaling (optional)
            '-window_title', self.windowTitle,             # give a window title
            '-hide_banner',
            # '-loglevel', 'verbose',
            '-nostats'
            ]

        # Start ffplay in a subprocess with a pipe for stdin
        self.ffplay_proc = subprocess.Popen(
            self.ffplay_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

    def showFrame(self, frame):
        """Display a frame on the screen.

        Args:
            frame (np.ndarray): A HxWx3 numpy array

        """
        if self.ffplay_proc is None:
            self.initialize(frame.shape[1], frame.shape[0])
        try:
            self.ffplay_proc.stdin.write(frame.tobytes())
            self.ffplay_proc.stdin.flush()
        except BrokenPipeError:
            debug('ffplayer: Broken pipe')
            debug(traceback.format_exc())
            self.close()
        except OSError:
            debug('ffplay: other error')
            debug(traceback.format_exc())
            self.close()
    def close(self, timeout=2):
        """Close the ffplay process
        """
        if self.ffplay_proc:
            if self.ffplay_proc.stdin:
                self.ffplay_proc.stdin.close()
            try:
                output, err = self.ffplay_proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                debug('ffplay: failed to close player gracefully:')
                debug(traceback.format_exc())
                output = b''
                err = b'Failed to communicate with ffplay process while closing.'
        else:
            output = ''
            err = ''
        self.ffplay_proc = None
        return output, err

if __name__ == "__main__":
    viewer = ffplayer(60, 'ffplayViewer.ffplayer example')
    try:
        k = 0
        while True:
            image = np.random.randint(0, 255, size=(1000, 1000, 3), dtype='uint8')
            if k < 50:
                viewer.showFrame(image)
            else:
                viewer.blank()
            time.sleep(1/30)
            k = k + 1
            k = k % 100
    except KeyboardInterrupt:
        output, err = viewer.close()
        debug('output')
        debug(output.decode('utf-8'))
        debug('err')
        debug(err.decode('utf-8'))
