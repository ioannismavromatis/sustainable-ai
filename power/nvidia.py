import time
from threading import Event, Thread

import pynvml

from utils import check_values, log

custom_logger = log.get_logger(__name__)
custom_logger = log.set_level(__name__, "info")


class NvidiaGPU(Thread):
    def __init__(self, sleep_time: int):
        Thread.__init__(self)
        self._stop_event = Event()
        self.sleep_time = check_values.set_time(sleep_time)
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()

        self.gpu_power_w = []
        self.gpu_temperature_c = []
        self.gpu_memory_free_b = []
        self.gpu_memory_used_b = []

    def __gpu_stats(self) -> None:
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            self.gpu_power_w.append(pynvml.nvmlDeviceGetPowerUsage(handle))
            self.gpu_temperature_c.append(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.gpu_memory_free_b.append(memory.free)
            self.gpu_memory_used_b.append(memory.used)

            custom_logger.info("Power: %s", pynvml.nvmlDeviceGetPowerUsage(handle))
            custom_logger.info(
                "Temperature: %s",
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            )
            custom_logger.info("Memory: %s", pynvml.nvmlDeviceGetMemoryInfo(handle))

    def reset(self):
        self.gpu_power_w = []
        self.gpu_temperature_c = []
        self.gpu_memory_free_b = []
        self.gpu_memory_used_b = []

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            self.__gpu_stats()
            time.sleep(self.sleep_time)
