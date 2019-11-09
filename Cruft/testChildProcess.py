import multiprocessing as mp
import queue
import requests
import time


class ChildProcess(mp.Process):
    def __init__(self, q, qout):
        super().__init__()
        self.qin = qin
        self.qout = qout
        self.daemon = True

    def run(self):
        while True:
            try:
                url = self.qin.get(block=False)
                r = requests.get(url, verify=False)
                self.qout.put(r.content)
                self.qin.task_done()
            except queue.Empty:
                break
            except requests.exceptions.RequestException as e:
                print(self.name, e)
                self.qin.task_done()
        print("Infinite loop terminates")

if __name__ == '__main__':
    qin = mp.JoinableQueue()
    qout = mp.Queue()
    for _ in range(5):
        qin.put('http://en.wikipedia.org')
    w = ChildProcess(qin, qout)
    w.start()
    qin.join()
    while True:
        try:
            qout.get(True, 0.1)         # Throw away remaining stuff in qout (or process it or whatever,
                                    # just get it out of the queue so the queue background process
                                    # can terminate, so your ChildProcess can terminate.
        except queue.Empty:
            break
    w.join()                # Wait for your ChildProcess to finish up.
    # time.sleep(1)         # Not necessary since we've joined the ChildProcess
    print(w.name, w.is_alive())
