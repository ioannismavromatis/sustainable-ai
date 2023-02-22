import threading
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataMonitor:
    cpu_energy_j: List[float] = field(default_factory=lambda: [])
    cpu_delta_power_w: List[float] = field(default_factory=lambda: [])
    cpu_percent: List[float] = field(default_factory=lambda: [])
    cpu_memory_percent: List[float] = field(default_factory=lambda: [])
    cpu_temperature_C: List[float] = field(default_factory=lambda: [])
    gpu_power_w: List[float] = field(default_factory=lambda: [])
    gpu_temperature_C: List[float] = field(default_factory=lambda: [])
    gpu_memory_free_B: List[float] = field(default_factory=lambda: [])
    gpu_memory_used_B: List[float] = field(default_factory=lambda: [])
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reset_values(self):
        with self.lock:
            for attr_name, attr_value in self.__dict__.items():
                if isinstance(attr_value, list):
                    setattr(self, attr_name, attr_value.__class__())

    def update_values_cpu(
        self, lists: Tuple[List[float], List[float], List[float], List[float]]
    ) -> None:
        with self.lock:
            self.cpu_energy_j.extend(lists[0])
            self.cpu_delta_power_w.extend(lists[1])
            self.cpu_percent.extend(lists[2])
            self.cpu_memory_percent.extend(lists[3])
            self.cpu_temperature_C.extend(lists[4])

    def update_values_gpu(
        self, lists: Tuple[List[float], List[float], List[float], List[float]]
    ) -> None:
        with self.lock:
            self.gpu_power_w.append(lists[0])
            self.gpu_temperature_C.append(lists[1])
            self.gpu_memory_free_B.append(lists[2])
            self.gpu_memory_used_B.append(lists[3])
