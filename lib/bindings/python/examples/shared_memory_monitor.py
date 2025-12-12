"""
Shared memory monitoring utilities for Dynamo components.
Enable with environment variable DYNAMO_MEMORY_PROFILE=1
"""

import gc
import os
import time

import psutil


class MemoryMonitor:
    """Memory monitor that can be shared across components"""

    def __init__(self, component_name):
        self.process = psutil.Process(os.getpid())
        self.request_count = 0
        self.start_time = time.time()
        self.component_name = component_name

    @property
    def initial_memory(self):
        if not hasattr(self, "_initial_memory"):
            self._initial_memory = self.process.memory_info().rss
        return self._initial_memory

    def log_memory(self, label=""):
        """Log current memory usage"""
        memory_info = self.process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
        rss_growth = (memory_info.rss - self.initial_memory) / 1024 / 1024
        uptime = time.time() - self.start_time

        print(
            f"[{self.component_name}] [{uptime:.1f}s] {label} Memory: RSS={rss_mb:.2f}MB (+{rss_growth:.2f}MB), VMS={vms_mb:.2f}MB, Requests={self.request_count}"
        )

    def increment_request(self):
        """Increment request counter and log if needed"""
        self.request_count += 1
        self.initial_memory

        # Log every 1000 requests
        if self.request_count % 1000 == 0:
            self.log_memory(f"After {self.request_count} requests:")

        # Force GC and log every 5000 requests
        if self.request_count % 5000 == 0:
            before_gc = self.process.memory_info().rss
            collected = gc.collect()
            after_gc = self.process.memory_info().rss
            freed_mb = (before_gc - after_gc) / 1024 / 1024
            print(
                f"[{self.component_name}]   GC collected {collected} objects, freed {freed_mb:.2f}MB"
            )


def create_monitor(component_name):
    """Create a memory monitor if profiling is enabled, otherwise return None"""
    if os.getenv("DYNAMO_MEMORY_PROFILE"):
        return MemoryMonitor(component_name)
    return None


def setup_background_monitor(monitor):
    """Create background monitoring task if monitor exists"""
    if not monitor:
        print("Memory monitoring not enabled.")
        return None

    import asyncio

    async def background_monitor():
        while True:
            await asyncio.sleep(15)  # Log every 15 seconds
            monitor.log_memory("Background check:")

    return asyncio.create_task(background_monitor())
