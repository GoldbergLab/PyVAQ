import multiprocessing as mp
import queue, time

class Worker(mp.Process):
    def __init__(self):
        super().__init__()
        self.q = mp.Queue()

    def run(self):
        print('worker starting work')
        for i in range(50000):
            self.q.put(i)
        print('worker done with work')
        self.q.close()
        self.q.join_thread()  # Good practice: join your own thread
        print('worker done!')

class Consumer(mp.Process):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def run(self):
        print('consumer starting')
        for i in range(5):
            item = self.q.get()
            print('consumer got:', item)
        # self.q.close()
        # self.q.join_thread()  # Good practice: join your own thread
        print('consumer done')

if __name__ == "__main__":
    # Parent code
    print('spawning children...')
    w = Worker()
    c = Consumer(w.q)
    c.start()
    w.start()

    print('parent monitoring...')
    while c.is_alive() and w.is_alive():
        time.sleep(0.1)
        print('consumer:', c.is_alive(), 'worker:', w.is_alive())

    print('clearing worker queue')
    while True:
        try:
            w.q.get(block=False)
        except queue.Empty:
            break

    print('joining worker')
    w.join()
    print('joining consumer')
    c.join()

    print('parent done')

# python "D:\Dropbox\Documents\Work\Cornell Lab Tech\Projects\Video VI\PyVAQ\Source\test\cancel_join_thread_test.py"
