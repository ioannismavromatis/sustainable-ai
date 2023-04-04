import time
from threading import Event, Thread

import pynvml

from utils import check_values, log

custom_logger = log.get_logger(__name__)
custom_logger = log.set_level(__name__, "info")


class NvidiaGPU(Thread):
    def __init__(self, sleep_time: int, data_monitor: object):
        Thread.__init__(self)
        self._stop_event = Event()
        self.sleep_time = check_values.set_time(sleep_time)
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()

        self.__initialize_attributes()
        self.data_monitor = data_monitor

    def __initialize_attributes(self) -> None:
        self._gpu_power_w: list[float] = []
        self._gpu_temperature_c: list[float] = []
        self._gpu_memory_free_b: list[float] = []
        self._gpu_memory_used_b: list[float] = []
        self._gpu_percent: list[float] = []

    def __gpu_stats(self) -> None:
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            self._gpu_power_w.append(pynvml.nvmlDeviceGetPowerUsage(handle))
            self._gpu_temperature_c.append(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self._gpu_memory_free_b.append(memory.free)
            self._gpu_memory_used_b.append(memory.used)
            percent = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self._gpu_percent.append(percent.gpu)

            custom_logger.debug("Power: %s", pynvml.nvmlDeviceGetPowerUsage(handle))
            custom_logger.debug(
                "Temperature: %s",
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            )
            custom_logger.debug("Memory: %s", pynvml.nvmlDeviceGetMemoryInfo(handle))
            custom_logger.debug(
                "GPU Percentage: %s%%", pynvml.nvmlDeviceGetUtilizationRates(handle)
            )

    def get_current_stats(self) -> None:
        values_to_save = (
            self._gpu_power_w,
            self._gpu_temperature_c,
            self._gpu_memory_free_b,
            self._gpu_memory_used_b,
            self._gpu_percent,
        )
        custom_logger.debug()
        self.data_monitor.update_values_gpu(values_to_save)
        self.reset()

    def reset(self) -> None:
        self.__initialize_attributes()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            self.__gpu_stats()
            time.sleep(self.sleep_time)
