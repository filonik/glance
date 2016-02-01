import time


class FrameTime(object):
    _seconds_per_frame = 1/30
    
    def __init__(self, frame):
        self.frame = frame
    
    def __call__(self):
        return self.frame * self._seconds_per_frame
    
    def __str__(self):
        return str(self.frame)


class Clock(object):
    def __init__(self, time):
        self.time = time
        self.reset()
    
    def reset(self):
        self.start_time = self.time()
        self.last_time = self.start_time
    
    def elapsed(self):
        return self.time() - self.start_time
    
    def tick(self):
        next_time = self.time()
        self.last_time, delta = next_time, next_time - self.last_time
        return delta
