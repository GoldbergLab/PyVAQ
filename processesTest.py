import multiprocessing as mp
import time

class testProcess(mp.Process):
    def __init__(self, id='', e=None, q=None, writer=False):
        mp.Process.__init__(self)
        self.a = 22
        self.e = e
        self.id = id
        self.q = q
        self.writer = writer

    def run(self):
        self.e.wait()
        print(self.id, "EVENT SET!")
        if self.writer:
            self.q.put("hello there!")
        else:
            print("GOT: ", self.q.get())


if __name__ == "__main__":
    e = mp.Event()
    q = mp.Queue()
    x = testProcess(id='x', e=e, q=q)
    y = testProcess(id='y', e=e, q=q, writer=True)
    del q
    time.sleep(1)

    x.start()
    y.start()
    input("Hit a key to continue")
    e.set()
