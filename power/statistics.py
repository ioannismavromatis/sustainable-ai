import threading
from dataclasses import dataclass, field


@dataclass
class DataMonitor:
    cpu_energy_j: list[float] = field(default_factory=lambda: [])
    cpu_delta_power_w: list[float] = field(default_factory=lambda: [])
    cpu_percent: list[float] = field(default_factory=lambda: [])
    cpu_memory_percent: list[float] = field(default_factory=lambda: [])
    cpu_temperature_c: list[float] = field(default_factory=lambda: [])
    gpu_power_w: list[float] = field(default_factory=lambda: [])
    gpu_temperature_c: list[float] = field(default_factory=lambda: [])
    gpu_memory_free_b: list[float] = field(default_factory=lambda: [])
    gpu_memory_used_b: list[float] = field(default_factory=lambda: [])
    gpu_percent: list[float] = field(default_factory=lambda: [])
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reset_values(self):
        with self.lock:
            for attr_name, attr_value in self.__dict__.items():
                if isinstance(attr_value, list):
                    setattr(self, attr_name, attr_value.__class__())

    def update_values_cpu(
        self, lists: tuple[list[float], list[float], list[float], list[float]]
    ) -> None:
        with self.lock:
            self.cpu_energy_j.extend(lists[0])
            self.cpu_delta_power_w.extend(lists[1])
            self.cpu_percent.extend(lists[2])
            self.cpu_memory_percent.extend(lists[3])
            self.cpu_temperature_c.extend(lists[4])

    def update_values_gpu(
        self,
        lists: tuple[list[float], list[float], list[float], list[float], list[float]],
    ) -> None:
        with self.lock:
            self.gpu_power_w.extend(lists[0])
            self.gpu_temperature_c.extend(lists[1])
            self.gpu_memory_free_b.extend(lists[2])
            self.gpu_memory_used_b.extend(lists[3])
            self.gpu_percent.extend(lists[4])

    def construct_results(self) -> dict:
        tmp_dict = {}
        with self.lock:
            for attr_name, attr_value in self.__dict__.items():
                if attr_name is not "lock" and (
                    attr_value or any(value != 0 for value in attr_value)
                ):
                    tmp_dict[attr_name] = attr_value

        return tmp_dict
