import multiprocessing as mp
import queue
import os
from pympler import tracker

class dummy:
    def __init__(self, k):
        self.k = k
        self.x = ['hi', 'ho']

class testProcess(mp.Process):
    def __init__(self):
        mp.Process.__init__(self, daemon=True)
        # self.msgQueue = mp.Queue()
        # self.PID = mp.Value('i', -1)
        self.memoryLeak = []

    def run(self):
        tr2 = tracker.SummaryTracker()
        # self.PID.value = os.getpid()
        k = 0
        print("Child tracker:")
        tr2.diff()
        while True:
            # try:
            #     msg = self.msgQueue.get(block=True, timeout=0.02)
            # except queue.Empty:
            #     msg = ''

            # if len(msg) > 0:
            #     print("Got message:")
            #     print(msg)
            #     if msg == "EXIT":
            #         break

            self.memoryLeak.append(dummy(k))
            # print("Leak size:", len(self.memoryLeak))
            k = k + 1
            if k == 100:
                tr2.print_diff()



if __name__ == "__main__":
    # tr = tracker.SummaryTracker()
    # print("Start:")
    # tr.print_diff()
    # msgQueue = mp.Queue()
    t = testProcess()
    t.start()
    import time
    time.sleep(5)
    # print("Post-init:")
    # tr.print_diff()
    # print("Memory leak?")
    # tr.print_diff()
