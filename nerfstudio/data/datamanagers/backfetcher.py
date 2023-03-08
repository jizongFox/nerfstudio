from __future__ import annotations

import typing
from functools import wraps, partial, lru_cache
from queue import Queue
from threading import Thread

if typing.TYPE_CHECKING:
    from .base_datamanager import VanillaDataManager


class background_fetch(Thread):

    def __init__(self, call_func=None, num_cache: int = 3) -> None:
        super().__init__()
        self.call_func = call_func
        self.index_queue = Queue()
        self.cache_queue = Queue()
        self._num_cache = num_cache
        self.daemon = True
        self.start()

    @lru_cache()
    def _init_workers(self):
        for i in range(self._num_cache):
            self.index_queue.put(i)

    def next_train(self, step):
        self.index_queue.put(step + self._num_cache)

        return self.cache_queue.get()

    def __call__(self, func):
        @wraps(func)
        def wrapper(_self_: "VanillaDataManager", step):
            self.call_func = partial(func, self=_self_)
            self._init_workers()
            return self.next_train(step)

        return wrapper

    def run(self) -> None:
        while True:
            index = self.index_queue.get()
            result = self.call_func(step=index)
            self.cache_queue.put(result, )
