#!/usr/bin/env python3
"""
benchmark_shared_image_ipc.py

Benchmark the throughput of SharedImageSender / SharedImageReceiver.

Run with default parameters:
    python benchmark_shared_image_ipc.py

Or try larger images / different buffer depths:
    python benchmark_shared_image_ipc.py --width 1920 --height 1080 --frames 2000 --buf 8
"""

import argparse
import ctypes
import multiprocessing as mp
import os
import sys
import time
from statistics import mean
import traceback
import queue
import ffmpegWriter as fw
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom

import numpy as np

from SharedImageQueue import SharedImageSender

def producer(sender: SharedImageSender, nframes: int, seed: int):
    """
    Fill the sender with nframes synthetic images as fast as possible.
    """
    rng = np.random.default_rng(seed)
    sender.setupBuffers()

    # Generate a random image
    loop_len = 10
    frames = rng.integers(
        0, 256, size=(sender.height // 32, sender.width // 16, sender.channels, loop_len), dtype=np.uint8
    )
    h, w, _, _ = frames.shape
    frames = zoom(frames, (sender.height / h, sender.width / w, 1, 1), order=0)
    # gaussian_filter(frames, (3, 3, 0, loop_len), output=frames)
    print('done creating synthetic data')
    sent = 0
    start = time.perf_counter()
    while sent < nframes:
        try:
            # print('sending {k} of {n}'.format(k=sent, n=nframes))
            sender.put(imarray=frames[:, :, :, sent % loop_len], metadata=None)  # fastest path
            sent += 1
        except queue.Full:
            continue
        except Exception as exc:
            print(f"[Producer] exception: {exc}", file=sys.stderr)
            traceback.print_exc()
            break
    elapsed = time.perf_counter() - start
    fps = nframes / elapsed
    print(f"[Producer] {nframes} frames in {elapsed:0.3f} s → {fps:0.1f} fps")


def consumer(receiver, nframes: int, shape: tuple):
    """
    Drain nframes from the receiver as fast as possible.
    """
    got = 0
    lag_samples = []
    writer = fw.ffmpegWriter('test.avi', "bytes", input_pixel_format='rgb24', fps=30, gpuVEnc=False)
    start = time.perf_counter()
    getTime = 0
    writeTime = 0
    while got < nframes-1:
        try:
            getStart = time.perf_counter()
            im = receiver.get()
            getTime += time.perf_counter() - getStart
            writeStart = time.perf_counter()
            writer.write(im, shape=shape)
            writeTime += time.perf_counter() - writeStart
            got += 1
            # print('got {n} frames'.format(n=got))
            lag_samples.append(receiver.qsize()[0])
        except queue.Empty:
            # Empty happens if the producer is slower for a moment
            continue
        except Exception as exc:
            print(f"[Consumer] exception: {exc}", file=sys.stderr)
            traceback.print_exc()
            break
    elapsed = time.perf_counter() - start
    writer.close()
    fps = got / elapsed
    avg_lag = mean(lag_samples) if lag_samples else 0
    print(f"[Consumer] {got} frames in {elapsed:0.3f} s → {fps:0.1f} fps "
          f"(avg queue depth {avg_lag:0.1f})")
    print("Get time: {g} Write time: {w}".format(g=getTime, w=writeTime))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640, help="image width")
    ap.add_argument("--height", type=int, default=480, help="image height")
    ap.add_argument("--channels", type=int, default=3, help="number of channels")
    ap.add_argument("--frames", type=int, default=1000, help="frames per run")
    ap.add_argument("--buf", type=int, default=4, help="ring-buffer slots")
    args = ap.parse_args()

    # Build the IPC primitive in *this* process
    sender = SharedImageSender(
        width=args.width,
        height=args.height,
        channels=args.channels,
        maxBufferSize=args.buf,
        allowOverflow=False,           # avoid hard stop when producer outruns consumer
        outputType="bytes",
        outputCopy=False,
        lockForOutput=False,
        verbose=0,
        name="bench",
    )

    receiver = sender.getReceiver()

    # Launch children
    cons = mp.Process(target=consumer, args=(receiver, args.frames, (args.height, args.width)))
    prod = mp.Process(target=producer, args=(sender, args.frames, os.getpid()))

    cons.start()
    prod.start()
    prod.join()
    cons.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")   # safer across platforms
    main()
