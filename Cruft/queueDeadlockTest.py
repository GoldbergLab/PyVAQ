import multiprocessing as mp
import time
import queue

class Producer(mp.Process):
    def __init__(self, queue, queue2):
        mp.Process.__init__(self)
        self.queue = queue
        self.queue2 = queue2
        self.stop = mp.Event()

    def stopProcess(self):
        # Method to shut down process gracefully
        self.stop.set()

    def run(self):
        k = 0
        while not self.stop.is_set():
            self.queue.put(k)
            self.queue2.put(k)
            k = k + 1
            time.sleep(0.1)
        print("Producer done")

class Consumer(mp.Process):
    def __init__(self, queue):
        mp.Process.__init__(self)
        self.queue = queue
        self.stop = mp.Event()

    def stopProcess(self):
        # Method to shut down process gracefully
        self.stop.set()

    def run(self):
        while not self.stop.is_set():
            try:
                received = self.queue.get(block=True, timeout=1.0)
                print("Received:", received)
            except queue.Empty:
                pass  # Queue empty
        print("Consumer done")

if __name__ == "__main__":
    q = mp.Queue()
    q2 = mp.Queue()
    p = Producer(q, q2)
    c = Consumer(q)
    c.start()
    p.start()
    input("Press any key to end.\n")
    print("Emptying q2:")
    while not q2.empty():
        print(q2.get())
    print("Done emptying q2")
    p.stopProcess()
    c.stopProcess()
    p.join()
    c.join()
