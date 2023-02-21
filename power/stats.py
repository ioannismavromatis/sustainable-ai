import os
import signal
import time
from threading import Event, Thread

import pandas as pd

import utils.log as logger
from utils import check_values, platform_info

from .generic_cpu import GenericCPU
from .intel import IntelCPU
from .nvidia import NvidiaGPU

custom_logger = logger.get_logger(__name__)
custom_logger = logger.set_level(__name__, "info")
custom_logger.debug("Logger initiated: %s", custom_logger)


class Stats(Thread):
    def __init__(
        self,
        sleep_time,
        net=None,
        run_id=0,
        file_dir="./results",
        file_name="stats.csv",
    ):
        Thread.__init__(self)
        self._stop_event = Event()
        self.run_id = check_values.set_id(run_id)
        self.sleep_time = check_values.set_time(sleep_time)
        self.net = net
        self.file_dir = file_dir
        self.file_name = file_name
        self.file_path = None

    def __cpu_monitor(self, platform, cpu_name, cpu_type="generic") -> None:
        if platform not in ["Darwin", "Linux", "Windows"]:
            raise ValueError(
                f"'platform must be 'Darwin', 'Linux', or 'Windows', now it is '{platform}"
            )
        if cpu_type not in ["Intel", "AMD", "M1", "generic"]:
            raise ValueError(
                f"'cpu_type must be 'Intel', 'AMD', 'M1', or 'generic', now it is '{cpu_type}"
            )

        if cpu_type == "Intel":
            self.intelCPU = IntelCPU(self.sleep_time)
            self.intelCPU.start()
        else:
            self.genericCPU = GenericCPU(self.sleep_time, cpu_name)
            self.genericCPU.start()

    def __get_cpu_stats(self) -> None:
        raise NotImplementedError

    def __gpu_monitor(self) -> None:
        self.nvidiaGPU = NvidiaGPU(self.sleep_time)
        self.nvidiaGPU.start()

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
        if self.net is None:
            custom_logger.critical("Network should not be None! Exiting program")
            os.kill(os.getpid(), signal.SIGINT)

        results_gpu_power_w = dict()
        results_gpu_temperature_C = dict()
        results_gpu_memory_free_B = dict()
        results_gpu_memory_used_B = dict()

        results_gpu_power_w["project_name"] = [self.__experiment_prefix(mode, "power")]
        results_gpu_power_w["epoch"] = [epoch]
        results_gpu_power_w["network"] = [self.net]
        results_gpu_power_w["values"] = [self.gpu_power_w]

        results_gpu_temperature_C["project_name"] = [
            self.__experiment_prefix(mode, "temp")
        ]
        results_gpu_temperature_C["epoch"] = [epoch]
        results_gpu_temperature_C["network"] = [self.net]
        results_gpu_temperature_C["values"] = [self.gpu_temperature_C]

        results_gpu_memory_free_B["project_name"] = [
            self.__experiment_prefix(mode, "memfree")
        ]
        results_gpu_memory_free_B["epoch"] = [epoch]
        results_gpu_memory_free_B["network"] = [self.net]
        results_gpu_memory_free_B["values"] = [self.gpu_memory_free_B]

        results_gpu_memory_used_B["project_name"] = [
            self.__experiment_prefix(mode, "memused")
        ]
        results_gpu_memory_used_B["epoch"] = [epoch]
        results_gpu_memory_used_B["network"] = [self.net]
        results_gpu_memory_used_B["values"] = [self.gpu_memory_used_B]

        return (
            results_gpu_power_w,
            results_gpu_temperature_C,
            results_gpu_memory_free_B,
            results_gpu_memory_used_B,
        )

    def __write_to_csv(self, mode, epoch):
        (
            results_gpu_power_w,
            results_gpu_temperature_C,
            results_gpu_memory_free_B,
            results_gpu_memory_used_B,
        ) = self.__construct_results_dict(mode, epoch)

        results_file = self.__find_file(mode)
        self.file_path = self.file_dir + "/" + results_file

        if not os.path.isfile(self.file_path):
            csv_file = open(self.file_path, "w+")
            pd.DataFrame(results_gpu_power_w).to_csv(self.file_path, index=False)
        else:
            csv_file = open(self.file_path, "a+")
            pd.DataFrame(results_gpu_power_w).to_csv(
                self.file_path, mode="a+", header=False, index=False
            )

        pd.DataFrame(results_gpu_temperature_C).to_csv(
            self.file_path, mode="a+", header=False, index=False
        )
        pd.DataFrame(results_gpu_memory_free_B).to_csv(
            self.file_path, mode="a+", header=False, index=False
        )
        pd.DataFrame(results_gpu_memory_used_B).to_csv(
            self.file_path, mode="a+", header=False, index=False
        )

        csv_file.close()

    def __stop_monitoring(self, system, chipset) -> None:
        raise NotImplementedError

    def get_results(self):
        return (
            self.gpu_power_w,
            self.gpu_temperature_C,
            self.gpu_memory_free_B,
            self.gpu_memory_used_B,
        )

    def save_results(self, mode, epoch):
        self.__write_to_csv(mode, epoch)

    def set_network(self, net):
        self.net = net

    def reset(self) -> None:
        self.gpu_power_w = []
        self.gpu_temperature_C = []
        self.gpu_memory_free_B = []
        self.gpu_memory_used_B = []

    def stop(self) -> None:
        self._stop_event.set()

    def run(self):
        system, cpu_name, _, chipset = platform_info.get_cpu_model()
        # self.__gpu_monitor()
        self.__cpu_monitor(platform=system, cpu_name=cpu_name, cpu_type=chipset)
        while not self._stop_event.is_set():
            # self.__get_cpu_stats()
            time.sleep(self.sleep_time)

        self.__stop_monitoring(system, chipset)
