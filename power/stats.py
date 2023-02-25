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
from .statistics import DataMonitor

custom_logger = logger.get_logger(__name__)
custom_logger = logger.set_level(__name__, "info")
custom_logger.debug("Logger initiated: %s", custom_logger)


class Stats(Thread):
    def __init__(
        self,
        sleep_time,
        device,
        net=None,
        run_id=0,
        file_dir="./results",
        file_name="stats.csv",
    ):
        Thread.__init__(self)
        self._stop_event = Event()
        self.run_id = check_values.set_id(run_id)
        self.sleep_time = check_values.set_time(sleep_time)
        self.device = device
        self.net = net
        self.file_dir = file_dir
        self.file_name = file_name
        self.file_path = None
        
        self.platform = platform_info.get_cpu_model()
        self.dataMonitor = DataMonitor()

    def __cpu_monitor(self) -> None:
        if self.platform['system_os'] not in ["Darwin", "Linux", "Windows"]:
            raise ValueError(
                f"'platform must be 'Darwin', 'Linux', or 'Windows', now it is '{ self.platform['system'] }"
            )
        if self.platform['chipset'] not in ["Intel", "AMD", "M1", "generic"]:
            raise ValueError(
                f"'cpu_type must be 'Intel', 'AMD', 'M1', or 'generic', now it is '{ self.platform['chipset'] }"
            )

        if self.platform['chipset'] == "Intel":
            self.intelCPU = IntelCPU(self.sleep_time, self.dataMonitor)
            self.intelCPU.start()
        else:
            self.genericCPU = GenericCPU(self.sleep_time, self.dataMonitor)
            self.genericCPU.start()

    def __get_cpu_stats(self) -> None:
        if self.platform['chipset'] == "Intel":
            self.intelCPU.get_current_stats()
        else:
            self.genericCPU.get_current_stats()

    def __get_gpu_stats(self) -> None:
        raise NotImplementedError

    def __gpu_monitor(self) -> None:
        if self.device == "cuda":
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

    def __return_monitors(self) -> list[str]:
        monitor_interfaces = []

        if self.platform['chipset'] == "Intel":
            monitor_interfaces.append(self.intelCPU)
        else:
            monitor_interfaces.append(self.genericCPU)

        if self.device == "cuda":
            monitor_interfaces.append(self.nvidiaGPU)

        return monitor_interfaces

    def __stop_monitoring(self) -> None:
        list_to_stop = self.__return_monitors(self.platform['chipset'])

        for device in list_to_stop:
            device.stop()

    def save_results(self, mode, epoch):
        self.__write_to_csv(mode, epoch)

    def set_network(self, net):
        self.net = net

    def reset(self) -> None:
        list_to_stop = self.__return_monitors()

        for device in list_to_stop:
            device.reset()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self):
        if self.device in ["cuda", "mps"]:
            self.__gpu_monitor()
        self.__cpu_monitor()
        while not self._stop_event.is_set():
            self.__get_cpu_stats()
            if self.device in ["cuda", "mps"]:
                self.__get_gpu_stats()
            time.sleep(2)

        self.__stop_monitoring()
