import multiprocessing as mp
class procchild(mp.Process):
    def __init__(self):
        mp.Process.__init__(self)
    def run(self):
        print("hi there, I am running")

class procchild2(procchild):
    def __init__(self):
            procchild.__init__(self)
    def run(self):
        procchild.run(self)
        print("I am child2 I am running")

if __name__ == "__main__":
    pc2 = procchild2()
    pc2.start()
