import multiprocessing as mp
import queue
import os

class testProcess(mp.Process):
    def __init__(self, msgQueue):
        mp.Process.__init__(self, daemon=True)
        self.msgQueue = msgQueue
        self.PID = mp.Value('i', -1)
        self.q2 = mp.Queue()

    def run(self):
        self.PID.value = os.getpid()
        while True:
            try:
                msg = self.msgQueue.get(block=True, timeout=0.2)
            except queue.Empty:
                msg = ''

            print("getting...")
            print("message:", self.q2.get(block=True))
            print('got')

            if len(msg) > 0:
                print("Got message:")
                print(msg)
                if msg == "EXIT":
                    break



if __name__ == "__main__":
    msgQueue = mp.Queue()
    t = testProcess(msgQueue)
    t.start()
    t.q2.put("Parent calling child")
    import time
    time.sleep(5)
