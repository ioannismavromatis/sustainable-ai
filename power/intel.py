import os
import re
import time
from threading import Event, Thread

import utils.log as logger
from utils import check_values, platform_info

custom_logger = logger.get_logger(__name__)
custom_logger = logger.set_level(__name__, "info")
custom_logger.debug("Logger initiated: %s", custom_logger)

RAPL_DIR = "/sys/class/powercap/"
CPU = 0
DRAM = 2


class IntelCPU(Thread):
    def __init__(self, sleep_time: int, dataMonitor: object):
        Thread.__init__(self)
        self._stop_event = Event()
        self.sleep_time = check_values.set_time(sleep_time)
        self.dataMonitor = dataMonitor

        self.energy_j_all = []
        self.delta_power_w_all = []
        self.cpu_percent_all = []
        self.memory_percent_all = []

        self._rapl_devices = []
        self._devices = []

        self.__get_rapl_devices()

    def __convert_rapl_name(self, name, pattern) -> str:
        if re.match(pattern, name):
            return "cpu:" + name[-1]

    def __get_rapl_devices(self) -> None:
        packages = list(filter(lambda x: ":" in x, os.listdir(RAPL_DIR)))
        parts_pattern = re.compile(r"intel-rapl:(\d):(\d)")
        devices_pattern = re.compile("intel-rapl:.")

        for package in packages:
            if re.fullmatch(devices_pattern, package):
                with open(os.path.join(RAPL_DIR, package, "name"), "r") as f:
                    name = f.read().strip()
                if name != "psys":
                    self._rapl_devices.append(package)
                    self._devices.append(
                        self.__convert_rapl_name(package, devices_pattern)
                    )

    def __read_energy(self, path) -> int:
        with open(os.path.join(path, "energy_uj"), "r") as f:
            return int(f.read())

    def __get_energy(self) -> None:
        cpu_energy_j = 0
        for package in self._rapl_devices:
            cpu_energy_j += self.__read_energy(os.path.join(RAPL_DIR, package))

        self.delta_power_w_all.append(self.__delta_power(cpu_energy_j))
        if self.energy_j_all:
            self.energy_j_all = [cpu_energy_j]
        else:
            self.energy_j_all.append(cpu_energy_j)

        custom_logger.debug("CPU energy consumption (mj): %s", cpu_energy_j)

    def __delta_power(self, last_measurement) -> int:
        if self.energy_j_all:
            joules = (last_measurement - self.energy_j_all[-1]) / 1000000
            watt = joules / self.sleep_time

            custom_logger.debug("CPU power consumption (w): %s", watt)
            return watt

        return 0

    def __get_utilisation(self) -> None:
        per_cpu, mem_usage = platform_info.cpu_utilisation()

        self.cpu_percent_all.append(sum(per_cpu) / len(per_cpu))
        self.memory_percent_all.append(mem_usage.percent)

    def reset(self) -> None:
        self.energy_j_all = []
        self.delta_power_w_all = []
        self.cpu_percent_all = []
        self.memory_percent_all = []

    def get_current_stats(self) -> None:
        values_to_save = (
            self.energy_j_all,
            self.delta_power_w_all,
            self.cpu_percent_all,
            self.memory_percent_all,
            self.cpu_temperature_all,
        )
        self.dataMonitor.update_values_cpu(values_to_save)
        self.reset()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self):
        try:
            for package in self._rapl_devices:
                os.stat(os.path.join(RAPL_DIR, package))

            while not self._stop_event.is_set():
                self.__get_utilisation()
                self.__get_energy()

                custom_logger.debug(
                    "CPU energy consumption (mj): %s", self.energy_j_all
                )
                custom_logger.debug("CPU power (w): %s", self.delta_power_w_all)
                custom_logger.debug("CPU utilisation (%%): %s", self.cpu_percent_all)
                custom_logger.debug(
                    "Memory utilisation (%%): %s", self.memory_percent_all
                )

                time.sleep(self.sleep_time)
        except PermissionError:
            custom_logger.critical("Provide SUDO rights to the script!!!!")
