import os
import re
import time
from threading import Event, Thread

from utils import check_values, log, platform_info

custom_logger = log.get_logger(__name__)
custom_logger = log.set_level(__name__, "info")

RAPL_DIR = "/sys/class/powercap/"
CPU = 0
DRAM = 2


class IntelCPU(Thread):
    def __init__(self, sleep_time: int, data_monitor: object):
        Thread.__init__(self)
        self._stop_event = Event()
        self.sleep_time = check_values.set_time(sleep_time)
        self.data_monitor = data_monitor

        self.__initialize_attributes()
        self._rapl_devices = []
        self._devices = []

        self.__get_rapl_devices()

    def __initialize_attributes(self) -> None:
        self._energy_j: list[float] = []
        self._delta_power_w: list[float] = []
        self._cpu_percent: list[float] = []
        self._memory_percent: list[float] = []
        self._cpu_temperature: list[float] = []

    def __convert_rapl_name(self, name, pattern) -> str:
        if re.match(pattern, name):
            return "cpu:" + name[-1]

    def __get_rapl_devices(self) -> None:
        packages = list(filter(lambda x: ":" in x, os.listdir(RAPL_DIR)))
        devices_pattern = re.compile("intel-rapl:.")

        for package in packages:
            if re.fullmatch(devices_pattern, package):
                with open(
                    os.path.join(RAPL_DIR, package, "name"), "r", encoding="utf-8"
                ) as file:
                    name = file.read().strip()
                if name != "psys":
                    self._rapl_devices.append(package)
                    self._devices.append(
                        self.__convert_rapl_name(package, devices_pattern)
                    )

    def __read_energy(self, path) -> int:
        with open(os.path.join(path, "energy_uj"), "r", encoding="utf-8") as file:
            return int(file.read())

    def __get_energy(self) -> None:
        cpu_energy_j = 0
        for package in self._rapl_devices:
            cpu_energy_j += self.__read_energy(os.path.join(RAPL_DIR, package))

        self._delta_power_w.append(self.__delta_power(cpu_energy_j))
        self._energy_j.append(cpu_energy_j)

        custom_logger.debug("CPU energy consumption (mj): %s", cpu_energy_j)

    def __delta_power(self, last_measurement) -> int:
        if self._energy_j:
            joules = (last_measurement - self._energy_j[-1]) / 1000000
            watt = joules / self.sleep_time

            custom_logger.debug("CPU power consumption (w): %s", watt)
            return watt

        return 0

    def __get_utilisation(self) -> None:
        per_cpu, mem_usage = platform_info.cpu_utilisation()

        self._cpu_percent.append(sum(per_cpu) / len(per_cpu))
        self._memory_percent.append(mem_usage.percent)

    def reset(self) -> None:
        self.__initialize_attributes()

    def get_current_stats(self) -> None:
        values_to_save = (
            self._energy_j,
            self._delta_power_w,
            self._cpu_percent,
            self._memory_percent,
            self._cpu_temperature,
        )
        self.data_monitor.update_values_cpu(values_to_save)
        self.reset()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self):
        if not os.geteuid() == 0:
            pass
            # raise PermissionError("Provide SUDO rights to the script!!!!")

        for package in self._rapl_devices:
            os.stat(os.path.join(RAPL_DIR, package))

        while not self._stop_event.is_set():
            self.__get_utilisation()
            self.__get_energy()

            time.sleep(self.sleep_time)
