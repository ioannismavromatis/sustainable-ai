import os
import signal
import time
from threading import Event, Thread

import pynvml

import utils.log as logger
from utils import check_values

custom_logger = logger.get_logger(__name__)
custom_logger = logger.set_level(__name__, "info")
custom_logger.debug("Logger initiated: %s", custom_logger)


class NvidiaGPU(Thread):
    def __init__(self, sleep_time: int):
        Thread.__init__(self)
        self._stop_event = Event()
        self.sleep_time = check_values.set_time(sleep_time)
        pynvml.nvmlInit()
        self.deviceCount = pynvml.nvmlDeviceGetCount()

        self.gpu_power_w = []
        self.gpu_temperature_C = []
        self.gpu_memory_free_B = []
        self.gpu_memory_used_B = []

    def __gpu_stats(self) -> None:
        for i in range(self.deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            self.gpu_power_w.append(pynvml.nvmlDeviceGetPowerUsage(handle))
            self.gpu_temperature_C.append(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.gpu_memory_free_B.append(memory.free)
            self.gpu_memory_used_B.append(memory.used)

            custom_logger.info("Power: %s", pynvml.nvmlDeviceGetPowerUsage(handle))
            custom_logger.info(
                "Temperature: %s",
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            )
            custom_logger.info("Memory: %s", pynvml.nvmlDeviceGetMemoryInfo(handle))

    def reset(self):
        self.gpu_power_w = []
        self.gpu_temperature_C = []
        self.gpu_memory_free_B = []
        self.gpu_memory_used_B = []

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            self.__gpu_stats()
            time.sleep(self.sleep_time)
