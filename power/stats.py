import os
import signal
import time
from threading import Event, Thread

import pandas as pd

import utils.log as logger

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
        self.run_id = run_id
        self.sleep_time = sleep_time
        self.net = net
        self.file_dir = file_dir
        self.file_name = file_name
        self.file_path = None

    def __generic_cpu(self) -> None:
        raise NotImplementedError

    def __cpu_stats(self, cpu_type="Intel", cpu_process="current") -> None:
        if cpu_process not in ["current", "all"]:
            raise ValueError(
                f"'cpu_process must be either 'current' or 'all', now it is '{cpu_process}"
            )

        if cpu_type not in ["Intel", "generic"]:
            raise ValueError(
                f"'cpu_type must be either 'Intel' or 'generic', now it is '{cpu_type}"
            )

        if cpu_type == "Intel":

            self.intelCPU = IntelCPU(self.sleep_time)
            self.intelCPU.start()
        else:
            self.__generic_cpu()

    def __get_cpu_stats(self) -> None:
        raise NotImplementedError

    def __gpu_stats(self) -> None:
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

    def reset(self):
        self.gpu_power_w = []
        self.gpu_temperature_C = []
        self.gpu_memory_free_B = []
        self.gpu_memory_used_B = []

    def stop(self):
        self._stop_event.set()

    def run(self):
        self.__gpu_stats()
        self.__cpu_stats(cpu_type="Intel", cpu_process="current")
        while not self._stop_event.is_set():
            # self.__get_cpu_stats()
            time.sleep(self.sleep_time)

        self.intelCPU.stop()
        self.nvidiaGPU.stop()
