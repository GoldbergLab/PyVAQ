import subprocess
import numpy as np
import time


class ffplayViewer:
    def __init__(self, maxFrameRate, windowTitle='ffplayViewer'):
        self.width = None
        self.height = None
        self.maxFrameRate = None
        self.windowTitle = windowTitle
        self.ffplay_cmd = None
        self.ffplay_proc = None

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

        self.ffplay_cmd = [
            'ffplay',
            '-f', 'rawvideo',
            '-pixel_format', 'rgb24',
            '-video_size', f'{self.width}x{self.height}',
            '-framerate', str(self.maxFrameRate*2),
            '-i', '-',
            # '-vf', f'drawtext=fontfile=\'C\\:/Windows/Fonts/Arial.ttf\':text={text}:fontcolor={color}:fontsize={fontsize}:x={textX}:y={textY}',  # ensure scaling (optional)
            '-window_title', self.windowTitle,             # give a window title
            '-hide_banner',
            '-loglevel', 'quiet',
            '-nostats'
            ]

        # Start ffplay in a subprocess with a pipe for stdin
        self.ffplay_proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)

    def showFrame(self, frame):
        """Display a frame on the screen.

        Args:
            frame (np.ndarray): A HxWx3 numpy array

        """
        if self.ffplay_proc is None:
            self.initialize(frame.shape[1], frame.shape[0])
        proc.stdin.write(frame.tobytes())
        proc.stdin.flush()

    def close(self):
        """Close the ffplay process
        """
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait()
