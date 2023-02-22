import difflib
import os
import time
from threading import Event, Thread

import pandas as pd

import utils.log as logger
from utils import check_values, platform_info

custom_logger = logger.get_logger(__name__)
custom_logger = logger.set_level(__name__, "info")
custom_logger.debug("Logger initiated: %s", custom_logger)

TDP_DATA = "./data/tdp.csv"
DEFAULT_TDP = 100


class GenericCPU(Thread):
    def __init__(self, sleep_time: int, cpu_name: str):
        Thread.__init__(self)
        self._stop_event = Event()
        self.sleep_time = check_values.set_time(sleep_time)
        self.cpu_name = cpu_name

        self._tdp = self.__find_tdp(TDP_DATA)

        self.energy_j_all = []
        self.delta_power_w_all = []
        self.cpu_percent_all = []
        self.memory_percent_all = []

    def __find_tdp(self, filepath) -> int:
        try:
            open(filepath)
        except FileNotFoundError:
            custom_logger.error(
                "Wrong path was given for the TDP values. Return default value."
            )
            return DEFAULT_TDP

        tdp_table = pd.read_csv(filepath)
        closest_match = difflib.get_close_matches(self.cpu_name, tdp_table.Name, n=1)[0]
        pd_list = tdp_table["Name"].str.strip().tolist()
        try:
            idx = pd_list.index(closest_match)
            tdp = tdp_table.loc[idx, "TDP"]
            custom_logger.info("Current TDP is: %s (W)", tdp)
            return tdp
        except ValueError:
            return DEFAULT_TDP

    def __get_energy(self) -> None:
        if len(self.cpu_percent_all) > 1:
            utilisation = (self.cpu_percent_all[-1] + self.cpu_percent_all[-2]) / 2
            delta_power = (utilisation / 100) * self._tdp / 3600 * self.sleep_time

            self.delta_power_w_all.append(delta_power)
            self.energy_j_all.append(self._tdp*(utilisation / 100)*1000*self.sleep_time)
            
            custom_logger.debug("Current power consumption: %s (W)", delta_power)
            custom_logger.debug("Current energy consumption: %s (mj)", self.energy_j_all[-1])

    def reset(self) -> None:
        self.energy_j_all = []
        self.delta_power_w_all = []
        self.cpu_percent_all = []
        self.memory_percent_all = []

    def __get_utilisation(self) -> None:
        per_cpu, mem_usage = platform_info.cpu_utilisation()

        self.cpu_percent_all.append(sum(per_cpu) / len(per_cpu))
        self.memory_percent_all.append(mem_usage.percent)

    def get_current_stats() -> None:
        raise NotImplementedError

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            self.__get_utilisation()
            self.__get_energy()
            time.sleep(self.sleep_time)
