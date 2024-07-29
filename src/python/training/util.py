import numpy as np
import hashlib
import json
import time


class Chronometer:

    def __init__(self, initial=0, paused=False):
        current_time = time.time()
        self.start_time = current_time - initial
        self.paused_time = 0
        self.paused = paused
        self.paused = paused
        self.paused_initial = current_time if paused else None
        self.last = initial

    def start(self, initial=0):
        if self.paused:
            self.paused_time += time.time() - self.paused_initial
            self.paused = False
        else:
            self.start_time = time.time() - initial
            self.paused_time = 0
            self.paused = False
            self.paused_initial = None

    def get(self):
        if self.paused:
            result = self.paused_initial - self.start_time - self.paused_time
        else:
            result = time.time() - self.start_time - self.paused_time
        self.last = result
        return result

    def pause(self):
        if not self.paused:
            self.paused_initial = time.time()
            self.paused = True

    def partial(self):
        last = self.last
        return self.get() - last


def hashing(dictionary):
    dictionary = {k: str(v) for k, v in dictionary.items() if v is not None}
    return hashlib.sha256(json.dumps(dictionary, sort_keys=True).encode()).hexdigest()


def weightprob(weight, abstolute=False):
    weight = np.array(weight, dtype=float)
    if abstolute:
        weight -= weight.min()
    n = len(weight)
    y = weight.sum()
    if y == 0:
        return np.full(n, 1 / n)
    p = 1 / (n * 10 ** 6)
    weight += p * y / (1 - p * n)
    return weight / weight.sum()
