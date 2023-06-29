import threading
import time
from dataclasses import dataclass, field


@dataclass
class DataMonitor:
    """
    A class for storing the monitoring data related to CPU, GPU, and RAM.

    Attributes:
        start_time (int): The start time of the monitoring.
        stop_time (int): The stop time of the monitoring.
        cpu_energy_uj (list[float]): Energy consumption of the CPU (in microjoules).
        cpu_delta_power_w (list[float]): Change in CPU power consumption (in watts).
        cpu_percent (list[float]): CPU utilization (in percentage).
        cpu_memory_percent (list[float]): Memory utilization percentage of the CPU (in percentage).
        cpu_temperature_c (list[float]): Temperature of the CPU (in degrees Celsius).
        gpu_power_w (list[float]): Power consumption of the GPU (in watts).
        gpu_temperature_c (list[float]): Temperature of the GPU (in degrees Celsius).
        gpu_memory_free_b (list[float]): Free memory of the GPU (in bytes).
        gpu_memory_used_b (list[float]): Used memory of the GPU (in bytes).
        gpu_percent (list[float]): GPU utilization (in percentage).
        ram_power_w (list[float]):Power consumption of the DRAM in watts.
        lock (threading.Lock): Thread lock for ensuring thread-safe operations.

    Methods:
        reset_values(): Resets all monitored values to empty lists.
        update_values_cpu(lists: tuple[list[float], list[float], list[float], list[float]]): Updates CPU-related values.
        update_values_gpu(lists: tuple[list[float], list[float], list[float], list[float], list[float]]): Updates GPU-related values.
        update_values_ram(lists: tuple[list[float]]): Updates RAM-related values.
        set_start_time(): Sets the start time of the monitoring.
        set_stop_time(): Sets the stop time of the monitoring.
        construct_results() -> dict: Constructs a dictionary containing the monitored data.

    """

    start_time: int = 0
    stop_time: int = 0
    cpu_energy_uj: list[float] = field(default_factory=lambda: [])
    cpu_delta_power_w: list[float] = field(default_factory=lambda: [])
    cpu_percent: list[float] = field(default_factory=lambda: [])
    cpu_memory_percent: list[float] = field(default_factory=lambda: [])
    cpu_temperature_c: list[float] = field(default_factory=lambda: [])
    gpu_power_w: list[float] = field(default_factory=lambda: [])
    gpu_temperature_c: list[float] = field(default_factory=lambda: [])
    gpu_memory_free_b: list[float] = field(default_factory=lambda: [])
    gpu_memory_used_b: list[float] = field(default_factory=lambda: [])
    gpu_percent: list[float] = field(default_factory=lambda: [])
    ram_power_w: list[float] = field(default_factory=lambda: [])
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reset_values(self):
        """
        Resets all monitored values to empty lists.
        """
        with self.lock:
            self.start_time = round(time.time_ns() / 1000000)
            for attr_name, attr_value in self.__dict__.items():
                if isinstance(attr_value, list):
                    setattr(self, attr_name, attr_value.__class__())

    def update_values_cpu(
        self, lists: tuple[list[float], list[float], list[float], list[float]]
    ) -> None:
        """
        Updates the CPU-related values with the provided lists.

        Args:
        - lists (tuple[list[float], list[float], list[float], list[float]]): A tuple containing lists of CPU-related values:
          - lists[0]: CPU energy consumption over time.
          - lists[1]: Change in CPU power consumption over time.
          - lists[2]: CPU utilization percentage over time.
          - lists[3]: Memory utilization percentage of the CPU over time.
        """
        with self.lock:
            self.cpu_energy_uj.extend(lists[0])
            self.cpu_delta_power_w.extend(lists[1])
            self.cpu_percent.extend(lists[2])
            self.cpu_memory_percent.extend(lists[3])
            self.cpu_temperature_c.extend(lists[4])

    def update_values_gpu(
        self,
        lists: tuple[list[float], list[float], list[float], list[float], list[float]],
    ) -> None:
        """
        Updates the GPU-related values with the provided lists.

        Args:
        - lists (tuple[list[float], list[float], list[float], list[float], list[float]]): A tuple containing lists of GPU-related values:
          - lists[0]: GPU power consumption over time.
          - lists[1]: GPU temperature over time.
          - lists[2]: Free memory of the GPU over time.
          - lists[3]: Used memory of the GPU over time.
          - lists[4]: GPU utilization percentage over time.
        """
        with self.lock:
            self.gpu_power_w.extend(lists[0])
            self.gpu_temperature_c.extend(lists[1])
            self.gpu_memory_free_b.extend(lists[2])
            self.gpu_memory_used_b.extend(lists[3])
            self.gpu_percent.extend(lists[4])

    def update_values_ram(self, lists: tuple[list[float]]) -> None:
        """
        Updates the RAM-related values with the provided list.

        Args:
        - lists (tuple[list[float]]): A tuple containing a list of RAM power consumption over time.
        """
        with self.lock:
            self.ram_power_w.append(lists[0])

    def set_start_time(self) -> None:
        """
        Sets the start time of the monitoring process.
        """
        self.start_time = round(time.time_ns() / 1000000)

    def set_stop_time(self) -> None:
        """
        Sets the stop time of the monitoring process.
        """
        self.stop_time = round(time.time_ns() / 1000000)

    def construct_results(self) -> dict:
        """
        Constructs a dictionary containing the monitored data.

        Returns:
        - dict: A dictionary containing the monitored data. The keys are the attribute names, and the values are the corresponding lists of values.
        """
        tmp_dict = {}
        if self.start_time > self.stop_time:
            raise ValueError("The stop time is older than the start time")

        with self.lock:
            for attr_name, attr_value in self.__dict__.items():
                if attr_name is not "lock" and (
                    attr_value or any(value != 0 for value in attr_value)
                ):
                    tmp_dict[attr_name] = attr_value

        return tmp_dict
