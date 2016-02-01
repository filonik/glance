import enum


class Status(enum.IntEnum):
    DeleteRequested = 0
    CreateRequested = 1
    UpdateRequested = 2
    Ready = 3
    Error = 4


class Resource(object):
    def __init__(self):
        super(Resource, self).__init__()

        self._status = Status.CreateRequested
        self._value = None

    def request_create(self): self._status = min(self._status, Status.CreateRequested)

    def request_update(self): self._status = min(self._status, Status.UpdateRequested)

    def request_delete(self): self._status = min(self._status, Status.DeleteRequested)

    def delete(self, *args, **kwargs):
        self._value = None
        return Status.CreateRequested

    def create(self, *args, **kwargs):
        return Status.UpdateRequested

    def update(self, *args, **kwargs):
        return Status.Ready if self._value else Status.Error

    def prepare(self, *args, **kwargs):
        if self._status is Status.Ready:
            return True

        if self._status is Status.DeleteRequested:
            self._status = self.delete(*args, **kwargs)
        if self._status is Status.CreateRequested:
            self._status = self.create(*args, **kwargs)
        if self._status is Status.UpdateRequested:
            self._status = self.update(*args, **kwargs)

        return self._status is Status.Ready


class Observable(object):
    def __init__(self):
        super(Observable, self).__init__()

        self._subscribers = []

    def subscribe(self, value):
        self._subscribers.append(value)

    def unsubscribe(self, value):
        self._subscribers.remove(value)

    def notify(self):
        for subscriber in self._subscribers:
            subscriber(self)
