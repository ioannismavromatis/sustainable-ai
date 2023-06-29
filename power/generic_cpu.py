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
        )[0]
        pd_list = tdp_table["Name"].str.strip().tolist()
        try:
            idx = pd_list.index(closest_match)
            tdp = tdp_table.loc[idx, "TDP"]
            custom_logger.info("Current TDP is: %s (W)", tdp)
            return tdp
        except ValueError:
            return DEFAULT_TDP

    def __get_energy(self) -> None:
        if len(self._cpu_percent) > 1:
            utilisation = sum(self._cpu_percent) / len(self._cpu_percent)
            delta_w = round(
                (utilisation / 100)
                * self._tdp
                / 3600
                * self.sleep_time
                * WATT_TO_MICROJOULE,
                8,
            )

            self._delta_power_w.append(delta_w)
            self._energy_j.append(
                round(
                    self._tdp
                    * (utilisation / 100)
                    * 1000
                    * self.sleep_time
                    * WATT_TO_MICROJOULE,
                    3,
                )
            )

            custom_logger.debug("Current power consumption: %s (W)", delta_w)
            custom_logger.debug(
                "Current energy consumption: %s (mj)", self._energy_j[-1]
            )

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
