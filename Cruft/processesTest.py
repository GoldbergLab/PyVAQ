import multiprocessing as mp
import time

class testProcess(mp.Process):
    def __init__(self, b, ID, ac, re, q=None):
        mp.Process.__init__(self)
        self.q = q
        self.b = b
        self.ID = ID
        self.ac = ac
        self.re = re

    def run(self):
        print(self.ID, "semaphore?")
        self.b.release()
        with self.ac.get_lock():
            self.ac.value += 1
        print(self.ID, "semaphore!")
        # time.sleep(0.5)
        # self.b.release()
        # with self.re.get_lock():
        #     self.re.value += 1


if __name__ == "__main__":
    b = mp.BoundedSemaphore(value=5)
    p = []
    ac = mp.Value('i', 0)
    re = mp.Value('i', 0)
    for k in range(100):
        p.append(testProcess(b, k, ac, re))
        p[-1].start()

    input("Hit a key to continue\n")

    print("Total acquires:", ac.value)
    print("Total releases:", re.value)
