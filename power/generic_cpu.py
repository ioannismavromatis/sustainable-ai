import difflib
import time
from threading import Event, Thread

import pandas as pd

from utils import check_values, log, platform_info

custom_logger = log.get_logger(__name__)
custom_logger = log.set_level(__name__, "info")

TDP_DATA = "./data/tdp.csv"
DEFAULT_TDP = 100

WATT_TO_MICROJOULE = 1000000


class GenericCPU(Thread):
    def __init__(self, sleep_time: int, data_monitor: object):
        Thread.__init__(self)
        self._stop_event = Event()
        self.sleep_time = check_values.set_time(sleep_time)
        self.data_monitor = data_monitor

        self.__initialize_attributes()
        self.platform = platform_info.get_cpu_model()
        self._tdp = self.__find_tdp(TDP_DATA)

    def __initialize_attributes(self) -> None:
        self._energy_j: list[float] = []
        self._delta_power_w: list[float] = []
        self._cpu_percent: list[float] = []
        self._memory_percent: list[float] = []
        self._cpu_temperature: list[float] = []

    def __find_tdp(self, filepath) -> int:
        try:
            tdp_table = pd.read_csv(filepath)
        except FileNotFoundError:
            custom_logger.error("Wrong path was given. Return default value.")
            return DEFAULT_TDP

        closest_match = difflib.get_close_matches(
            self.platform["cpu_name"], tdp_table.Name, n=1
        )

        if not closest_match:
            custom_logger.warning(
                "TDP not found. Default value used: %s (W)", DEFAULT_TDP
            )
            return DEFAULT_TDP

        closest_match = closest_match[0]
        mask = tdp_table.Name == closest_match
        index_of_closest_match = tdp_table[mask].index[0]

        tdp = tdp_table.loc[index_of_closest_match, "TDP"]
        custom_logger.info("Current TDP is: %s (W)", tdp)

        return tdp

    def __get_energy(self) -> None:
        if len(self._cpu_percent) > 1:
            utilisation = self._cpu_percent[-1]

            delta_w = self._tdp * (utilisation / 100.0)
            self._delta_power_w.append(delta_w)

            custom_logger.debug("Power consumption: %s (W)", delta_w)

            if not self._energy_j:
                self._energy_j.append(round(delta_w * WATT_TO_MICROJOULE * self.sleep_time))
            else:
                self._energy_j.append(
                    self._energy_j[-1] + round(delta_w * WATT_TO_MICROJOULE * self.sleep_time)
                )

            custom_logger.debug("Energy consumption: %s (uj)", self._energy_j[-1])

    def reset(self) -> None:
        self.__initialize_attributes()

    def __get_utilisation(self) -> None:
        per_cpu, mem_usage = platform_info.cpu_utilisation()

        self._cpu_percent.append(round(sum(per_cpu) / len(per_cpu), 1))
        self._memory_percent.append(mem_usage.percent)

    def __get_temperature(self) -> None:
        current_temp = platform_info.cpu_temperature(self.platform["system_os"])

        self._cpu_temperature.append(current_temp)

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
        while not self._stop_event.is_set():
            self.__get_utilisation()
            self.__get_energy()
            self.__get_temperature()
            time.sleep(self.sleep_time)
