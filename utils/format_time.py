import re


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def plot_time(time_str):
    numbers_only = re.findall(r"\d+", time_str)
    characters_only = re.split(r"\d+", time_str)
    characters_only = [x for x in characters_only if x != ""]
    final_ms = 0
    for idx, number in enumerate(numbers_only):
        if characters_only[idx] == "D":
            final_ms = final_ms + int(number) * 86400000
        elif characters_only[idx] == "h":
            final_ms = final_ms + int(number) * 3600000
        elif characters_only[idx] == "m":
            final_ms = final_ms + int(number) * 60000
        elif characters_only[idx] == "s":
            final_ms = final_ms + int(number) * 1000
        elif characters_only[idx] == "ms":
            final_ms = final_ms + int(number)
        else:
            raise ValueError("Invalid time format")

    return final_ms
