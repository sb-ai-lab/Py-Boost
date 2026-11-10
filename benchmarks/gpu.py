"""Measurement helpers. GPU timing and allocation tracking run separately."""

import threading


def memory_tracker(cp):
    class Tracker(cp.cuda.MemoryHook):
        def __init__(self):
            self.live = {}
            self.current = 0
            self.peak = 0

        def malloc_postprocess(self, device_id, size, mem_size, mem_ptr, pmem_id):
            if mem_ptr and pmem_id:
                key = (device_id, pmem_id)
                self.current += mem_size - self.live.get(key, 0)
                self.live[key] = mem_size
                self.peak = max(self.peak, self.current)

        def free_postprocess(self, device_id, mem_size, mem_ptr, pmem_id):
            self.current -= self.live.pop((device_id, pmem_id), 0)

        def reset_peak(self):
            self.peak = self.current

    return Tracker()


class RssPeak:
    """Sample absolute RSS during one phase; includes resident CPU input data."""

    def __init__(self, process):
        self.process = process
        self.stop = threading.Event()

    def __enter__(self):
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(.005):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *exc):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()
