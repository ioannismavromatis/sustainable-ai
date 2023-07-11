import json
import os
import signal
import time
from threading import Event, Thread

from utils import check_values, log, platform_info

from .generic_cpu import GenericCPU
from .intel import IntelCPU
from .nvidia import NvidiaGPU
from .ram import RAM
from .statistics import DataMonitor

custom_logger = log.get_logger(__name__)
custom_logger = log.set_level(__name__, "info")


class Stats(Thread):
    def __init__(
        self,
        sleep_time,
        device,
        generic_cpu=False,
        net=None,
        run_id=0,
        file_dir="./results",
    ):
        Thread.__init__(self)
        self._stop_event = Event()
        self.run_id = check_values.set_id(run_id)
        self.sleep_time = check_values.set_time(sleep_time)
        self.device = str(device)  # device returned from PyTorch is an object
        self.generic_cpu = generic_cpu
        self.net = net
        self.file_dir = file_dir
        self.file_path = None

        self.platform = platform_info.get_cpu_model()
        self.data_monitor = DataMonitor()

        self.__gpu_monitor()
        self.__cpu_monitor()
        self.__ram_monitor()

    def __cpu_monitor(self) -> None:
        if self.platform["system_os"] not in ["Darwin", "Linux", "Windows"]:
            raise ValueError(
                f"'platform must be 'Darwin', 'Linux', or 'Windows', now it is '{ self.platform['system'] }"
            )
        if self.platform["chipset"] not in ["Intel", "AMD", "M1", "generic"]:
            raise ValueError(
                f"'cpu_type must be 'Intel', 'AMD', 'M1', or 'generic', now it is '{ self.platform['chipset'] }"
            )

        if self.platform["chipset"] == "Intel":
            self.intel_cpu = IntelCPU(self.sleep_time, self.data_monitor)
            if self.intel_cpu.rapl_devices_exist() and self.generic_cpu is False:
                self.intel_cpu.start()
                custom_logger.info("Intel CPU with RAPL support is detected.")
            else:
                if self.generic_cpu is False:
                    custom_logger.warning("Intel CPU without RAPL support is detected.")
                custom_logger.info("Defaulting to generic CPU.")
                self.platform["chipset"] = "generic"
                self.generic_cpu = GenericCPU(self.sleep_time, self.data_monitor)
                self.generic_cpu.start()
        else:
            self.generic_cpu = GenericCPU(self.sleep_time, self.data_monitor)
            self.generic_cpu.start()

    def __get_cpu_stats(self) -> None:
        if self.platform["chipset"] == "Intel":
            self.intel_cpu.get_current_stats()
        else:
            self.generic_cpu.get_current_stats()

    def __ram_monitor(self) -> None:
        self.ram = RAM(self.sleep_time, self.data_monitor)
        self.ram.start()

    def __get_ram_stats(self) -> None:
        self.ram.get_current_stats()

    def __gpu_monitor(self) -> None:
        if self.device == "cuda":
            self.nvidia_gpu = NvidiaGPU(self.sleep_time, self.data_monitor)
            self.nvidia_gpu.start()

    def __get_gpu_stats(self) -> None:
        self.nvidia_gpu.get_current_stats()

    def __experiment_prefix(self):
        return "exp" + "_" + str(self.run_id)

    def __find_file(self, mode):
        file_name = mode + ".json"
        if "train" in mode or "test" in mode:
            return file_name
        else:
            raise ValueError(
                "A wrong mode type was given. Give either 'train' or 'test'."
            )

    def __construct_results_dict(self, results_dict, epoch) -> dict:
        if self.net is None:
            custom_logger.critical("Network should not be None! Exiting program")
            os.kill(os.getpid(), signal.SIGINT)

        experiment = self.__experiment_prefix()
        if experiment not in results_dict:
            results_dict[experiment] = dict()
        if self.net not in results_dict[experiment]:
            results_dict[experiment][self.net] = dict()

        tmp_results = self.data_monitor.construct_results()
        results_dict[experiment][self.net][epoch] = tmp_results

        return results_dict

    def __write_to_json(self, mode, epoch) -> None:
        results_file = self.__find_file(mode)
        self.file_path = self.file_dir + "/" + results_file

        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as json_file:
                results_dict = json.load(json_file)
        else:
            results_dict = dict()

        csv_data = self.__construct_results_dict(results_dict, epoch)
        with open(self.file_path, "w+", encoding="utf-8") as file:
            json.dump(csv_data, file)

    def __return_monitors(self) -> list[str]:
        monitor_interfaces = []

        if self.platform["chipset"] == "Intel":
            monitor_interfaces.append(self.intel_cpu)
        else:
            monitor_interfaces.append(self.generic_cpu)

        if self.device == "cuda":
            monitor_interfaces.append(self.nvidia_gpu)

        monitor_interfaces.append(self.ram)

        return monitor_interfaces

    def __stop_monitoring(self) -> None:
        list_to_stop = self.__return_monitors()

        for device in list_to_stop:
            device.stop()

    def save_results(self, mode, epoch) -> None:
        self.data_monitor.set_stop_time()
        self.__write_to_json(mode, epoch)
        self.data_monitor.reset_values()

    def set_network(self, net) -> None:
        self.net = net

    def reset(self) -> None:
        list_to_stop = self.__return_monitors()

        for device in list_to_stop:
            device.reset()
        self.data_monitor.set_start_time()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            self.__get_cpu_stats()
            self.__get_ram_stats()
            if self.device in ["cuda"]:
                self.__get_gpu_stats()
            time.sleep(1)

        self.__stop_monitoring()
