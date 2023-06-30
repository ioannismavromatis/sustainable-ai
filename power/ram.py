import re, os
import time
from threading import Event, Thread
from pathlib import Path

from utils import check_values, log

custom_logger = log.get_logger(__name__)
custom_logger = log.set_level(__name__, "info")

AVG_AMPS_PER_DIMM = 1.3


class RAM(Thread):
    def __init__(self, sleep_time: int, data_monitor: object):
        Thread.__init__(self)
        self._stop_event = Event()
        self.sleep_time = check_values.set_time(sleep_time)
        self.data_monitor = data_monitor
        self.dimm_count, self.dimm_size, self.voltage = self.__get_dram_dimms()
        self._ram_power = self.__calculate_power()

        self.__initialize_attributes()

    def __initialize_attributes(self) -> None:
        self._ram_power_w: list[float] = []

    def __get_dram_dimms(self):
        ROOT_DIR = os.path.dirname(Path(__file__).parent)
        file_path = ROOT_DIR + "/results/meminfo.txt"
        if not os.path.isfile(file_path):
            raise ValueError("Run 'dmidecode -t memory' and save the output to './results/meminfo.txt'")

        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()

        p = re.compile(r"\sSpeed:\s(\d+)\sMT/s")
        dimm_count = sum(1 for x in file_content.splitlines() if p.match(x))

        dimm_size = []
        p = re.compile(r"\sSize:\s(\d+)\s.B")
        for line in file_content.splitlines():
            if p.match(line):
                split_string = p.match(line).group(0).split()
                if "MB" in split_string:
                    dimm_size.append(int(p.match(line).group(1)) / 1024)
                elif "GB" in split_string:
                    dimm_size.append(p.match(line).group(1))
                else:
                    raise ValueError("Unknown memory size")

        voltage = []
        p = re.compile(r"\sConfigured\sVoltage:\s(\d+.\d+)\sV")
        for line in file_content.splitlines():
            if p.match(line):
                voltage.append(p.match(line).group(1))

        return dimm_count, dimm_size, voltage

    def __calculate_power(self):
        voltage = [float(v) for v in self.voltage]
        power_per_eight_gb = sum(v * AVG_AMPS_PER_DIMM for v in voltage)
        total_power = sum(map(int, self.dimm_size)) / power_per_eight_gb
        return total_power

    def __get_energy(self) -> None:
        self._ram_power_w.append(round(self._ram_power, 2))

    def reset(self) -> None:
        self.__initialize_attributes()

    def get_current_stats(self) -> None:
        values_to_save = self._ram_power_w
        self.data_monitor.update_values_ram(values_to_save)
        self.reset()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            self.__get_energy()

            time.sleep(self.sleep_time)
