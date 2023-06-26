import time, re
import subprocess
from threading import Event, Thread

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
        output = subprocess.check_output(["sudo", "dmidecode", "-t", "memory"], universal_newlines=True)
        
        p = re.compile("\sSpeed:\s(\d+)\sMT/s")
        dimm_count = sum(1 for x in output.splitlines() if p.match(x))
        
        dimm_size = []
        p = re.compile("\sSize:\s(\d+)\sMB")
        for x in output.splitlines():
            if p.match(x):
                dimm_size.append(p.match(x).group(1))

        voltage = []
        p = re.compile("\sConfigured\sVoltage:\s(\d+.\d+)\sV")
        for x in output.splitlines():
            if p.match(x):
                voltage.append(p.match(x).group(1))
        
        return dimm_count, dimm_size, voltage

    def __calculate_power(self):
        voltage = [float(v) for v in self.voltage]
        power_per_eight_gb = sum(v * AVG_AMPS_PER_DIMM for v in voltage)
        total_power =  sum(map(int, self.dimm_size)) / 1000 / power_per_eight_gb
        return total_power
    
    def __get_energy(self) -> None:
        self._ram_power_w.append(self._ram_power)
        
    def reset(self) -> None:
        self.__initialize_attributes()
        
    def get_current_stats(self) -> None:
        values_to_save = (self._ram_power_w)
        self.data_monitor.update_values_ram(values_to_save)
        self.reset()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            self.__get_energy()

            time.sleep(self.sleep_time)
