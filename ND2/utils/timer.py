import time
import logging
import numpy as np
from functools import wraps
from collections import defaultdict

logger = logging.getLogger('ND2.timer')

class WrapTimer:
    def __init__(self):
        self.timeit_record = defaultdict(lambda: np.zeros(4)) # 1, T, T^2, t
    
    def timeit(self, report_time=3600.0):
        """ Add `@wraptimer.timeit(report_time=3600)` before a function to record its time """
        def decorate(f):
            @wraps(f)
            def inner(*args, **kwargs):
                start = time.time()
                ret = f(*args, **kwargs)
                end = time.time()
                dt = end - start
                self.timeit_record[f.__name__][0] += 1
                self.timeit_record[f.__name__][1] += dt
                self.timeit_record[f.__name__][2] += dt ** 2 
                
                if report_time > 0 and self.timeit_record[f.__name__][1] > self.timeit_record[f.__name__][3] + report_time:
                    t0, t1, t2, _ = self.timeit_record[f.__name__]
                    t_mean = t1 / t0
                    t_std = np.sqrt(t2 / t0 - t_mean ** 2)
                    logger.info(f'{f.__qualname__}: last={dt*1e3:.1f}ms, count={t0}, mean={t_mean*1e3:.1f}±{t_std*1e3:.1f}ms, total={t1:.1f}s')
                    self.timeit_record[f.__name__][3] = self.timeit_record[f.__name__][1]
                return ret
            return inner
        return decorate

    def conclude_timeit(self, clear=True):
        conclusion = []
        for f in self.timeit_record:
            t0, t1, t2, _ = self.timeit_record[f]
            t_mean = t1 / t0
            t_std = np.sqrt(t2 / t0 - t_mean ** 2)
            conclusion.append((f'{f}: count={t0}, mean={t_mean*1e3:.1f}±{t_std*1e3:.1f}ms, total={t1:.1f}s', t1))
        conclusion = sorted(conclusion, key=lambda x: x[1], reverse=True)
        logger.info('Timeit Conclusion:\n' + '\n'.join([c[0] for c in conclusion]) + '\n')
        if clear: self.timeit_record.clear()

wraptimer = WrapTimer()


class Timer:
    def __init__(self):
        self.count = 0
        self.time = 0
        self.start_time = time.time()

    def add(self, n):
        self.count += n
        self.time += time.time() - self.start_time
        self.start_time = time.time()

    def pop(self, reset=True):
        speed = self.count / (self.time + 1e-9)
        if reset: self.count = self.time = 0
        return speed
    
    def reset(self):
        self.pop(reset=True)

class AbsTimer(Timer):
    def __init__(self):
        self.last = 0
        self.count = 0
        self.time = 0
        self.start_time = time.time()

    def set(self, n):
        self.count = n
        self.time += time.time() - self.start_time
        self.start_time = time.time()

    def pop(self, reset=True):
        speed = (self.count - self.last) / (self.time + 1e-9)
        if reset: 
            self.last = self.count
            self.time = 0
        return speed

class CycleTimer(Timer):
    def __init__(self, N):
        self.N = N
        self.time = [0] * N
        self.start_time = time.time()

    def add(self, idx):
        self.time[idx] += time.time() - self.start_time
        self.start_time = time.time()

    def pop(self, reset=True):
        total = sum(self.time) + 1e-9
        ratio = [t / total for t in self.time]
        if reset: self.time = [0] * self.N
        return total, ratio


class NamedTimer(Timer):
    def __init__(self):
        self.time = {}
        self.count = {}
        self.start_time = time.time()

    def add(self, name):
        if name not in self.time: self.time[name] = self.count[name] = 0
        self.time[name] += time.time() - self.start_time
        self.count[name] += 1
        self.start_time = time.time()

    def pop(self, reset=True, raw=False):
        time, count = self.time, self.count
        if reset: 
            self.time, self.count = {}, {}
        if raw: 
            return time, count
        total = sum(time.values()) + 1e-9
        tmp = []
        for k in time.keys():
            if count[k] == 0: continue
            if count[k] == 1: tmp.append(f'{k}={time[k]*1000:.0f}ms ({time[k]/total:.0%})')
            else: tmp.append(f'{k}={count[k]}*{time[k]/count[k]*1000:.0f}ms ({time[k]/total:.0%})')
        return ', '.join(tmp)

    def total(self):
        return sum(self.time.values())