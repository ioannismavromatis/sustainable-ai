import platform
import subprocess

import psutil

from utils import log

custom_logger = log.get_logger(__name__)
custom_logger = log.set_level(__name__, "info")


def get_cpu_model() -> dict:
    system = platform.system()
    if system == "Windows":
        arch = platform.machine()
        cpu_name = (
            subprocess.check_output("wmic cpu get name").decode().strip().split("\n")[1]
        )
    elif system == "Darwin":
        arch = subprocess.check_output(["uname", "-m"]).decode().strip()
        cpu_name = (
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            .decode()
            .strip()
        )
    elif system == "Linux":
        arch = platform.machine()
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.strip().startswith("model name"):
                    cpu_name = line.split(":")[1].strip()
    else:
        return None

    if "Intel" in cpu_name:
        chipset = "Intel"
    elif "AMD" in cpu_name:
        chipset = "AMD"
    elif "Apple M1" in cpu_name:
        chipset = "M1"
    else:
        chipset = "generic"

    return {"system_os": system, "cpu_name": cpu_name, "arch": arch, "chipset": chipset}


def cpu_utilisation():
    per_cpu = psutil.cpu_percent(percpu=True)
    mem_usage = psutil.virtual_memory()

    custom_logger.debug("Core Number: %s", psutil.cpu_count())
    custom_logger.debug("Core Utilisation: %s", per_cpu)
    custom_logger.debug("RAM Free (%%): %s", mem_usage.percent)
    custom_logger.debug("RAM Total (G): %s", mem_usage.total / (1024**3))
    custom_logger.debug("RAM Used (G): %s", mem_usage.used / (1024**3))

    return per_cpu, mem_usage


def cpu_temperature(platform):
    if platform != "Linux":
        return 0
        # raise NotImplementedError("CPU temperature not available on this system")

    temperatures = psutil.sensors_temperatures()
    if "coretemp" in temperatures:
        cpu_temperatures = temperatures["coretemp"]
        max_temp = max([temp.current for temp in cpu_temperatures])

        custom_logger.debug("CPU temperature: %s", max_temp)
        return max_temp

    return 0
