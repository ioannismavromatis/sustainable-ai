import platform
import subprocess


def get_cpu_model():
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

    return system, cpu_name, arch, chipset
