import os
import time
from threading import Thread

import pandas as pd

import utils.log as logger

custom_logger = logger.get_logger(__name__)
custom_logger = logger.set_level(__name__, "info")
custom_logger.debug("Logger initiated: %s", custom_logger)


class Stats(Thread):
    def __init__(
        self,
        pynvml,
        deviceCount,
        sleep_time,
        net,
        run_id=0,
        file_dir="./results",
        file_name="stats.csv",
    ):
        Thread.__init__(self)
        self.daemon = False
        self.run_id = run_id

        self.pynvml = pynvml
        self.deviceCount = deviceCount
        self.sleep_time = sleep_time
        self.net = net
        self.file_dir = file_dir
        self.file_name = file_name
        self.file_path = None
        self.power_w = []
        self.temperature_C = []
        self.memory_free_B = []
        self.memory_used_B = []

    def __gpu_stats(self) -> None:
        for i in range(self.deviceCount):
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)

            self.power_w.append(self.pynvml.nvmlDeviceGetPowerUsage(handle))
            self.temperature_C.append(
                self.pynvml.nvmlDeviceGetTemperature(
                    handle, self.pynvml.NVML_TEMPERATURE_GPU
                )
            )
            memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.memory_free_B.append(memory.free)
            self.memory_used_B.append(memory.used)

            custom_logger.debug(
                "Power: %s", self.pynvml.nvmlDeviceGetPowerUsage(handle)
            )
            custom_logger.debug(
                "Temperature: %s",
                self.pynvml.nvmlDeviceGetTemperature(
                    handle, self.pynvml.NVML_TEMPERATURE_GPU
                ),
            )
            custom_logger.debug(
                "Memory: %s", self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            )

    def __experiment_prefix(self, mode, array_name):
        return (
            mode + "-" + array_name + "-" + str(self.run_id)
            if self.run_id != 0
            else mode + "-" + array_name
        )

    def __find_file(self, mode):
        tmp_list = self.file_name.split(".")
        tmp_list[0] = tmp_list[0] + "_" + mode
        results_file = ".".join(map(str, tmp_list))
        if "train" in mode or "test" in mode:
            return results_file
        else:
            raise ValueError(
                "A wrong mode type was given. Give either 'train' or 'test'."
            )

    def __construct_results_dict(self, mode, epoch):
        results_power_w = dict()
        results_temperature_C = dict()
        results_memory_free_B = dict()
        results_memory_used_B = dict()

        results_power_w["project_name"] = [self.__experiment_prefix(mode, "power")]
        results_power_w["epoch"] = [epoch]
        results_power_w["network"] = [self.net]
        results_power_w["values"] = [self.power_w]

        results_temperature_C["project_name"] = [self.__experiment_prefix(mode, "temp")]
        results_temperature_C["epoch"] = [epoch]
        results_temperature_C["network"] = [self.net]
        results_temperature_C["values"] = [self.temperature_C]

        results_memory_free_B["project_name"] = [
            self.__experiment_prefix(mode, "memfree")
        ]
        results_memory_free_B["epoch"] = [epoch]
        results_memory_free_B["network"] = [self.net]
        results_memory_free_B["values"] = [self.memory_free_B]

        results_memory_used_B["project_name"] = [
            self.__experiment_prefix(mode, "memused")
        ]
        results_memory_used_B["epoch"] = [epoch]
        results_memory_used_B["network"] = [self.net]
        results_memory_used_B["values"] = [self.memory_used_B]

        return (
            results_power_w,
            results_temperature_C,
            results_memory_free_B,
            results_memory_used_B,
        )

    def __write_to_csv(self, mode, epoch):
        (
            results_power_w,
            results_temperature_C,
            results_memory_free_B,
            results_memory_used_B,
        ) = self.__construct_results_dict(mode, epoch)

        results_file = self.__find_file(mode)
        self.file_path = self.file_dir + "/" + results_file

        if not os.path.isfile(self.file_path):
            csv_file = open(self.file_path, "w+")
            pd.DataFrame(results_power_w).to_csv(self.file_path, index=False)
        else:
            csv_file = open(self.file_path, "a+")
            pd.DataFrame(results_power_w).to_csv(
                self.file_path, mode="a+", header=False, index=False
            )

        pd.DataFrame(results_temperature_C).to_csv(
            self.file_path, mode="a+", header=False, index=False
        )
        pd.DataFrame(results_memory_free_B).to_csv(
            self.file_path, mode="a+", header=False, index=False
        )
        pd.DataFrame(results_memory_used_B).to_csv(
            self.file_path, mode="a+", header=False, index=False
        )

        csv_file.close()

    def get_results(self):
        return (
            self.power_w,
            self.temperature_C,
            self.memory_free_B,
            self.memory_used_B,
        )

    def save_results(self, mode, epoch):
        self.__write_to_csv(mode, epoch)

    def reset(self):
        self.power_w = []
        self.temperature_C = []
        self.memory_free_B = []
        self.memory_used_B = []

    def run(self):
        while True:
            self.__gpu_stats()
            time.sleep(self.sleep_time)
