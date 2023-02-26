import os

import pandas as pd

from utils import log

custom_logger = log.get_logger(__name__)
custom_logger = log.set_level(__name__, "info")


class ToolResults:
    def __init__(self, net, file_dir="./results", file_name="mlresults.csv", run_id=0):
        self.project_name = net
        self.net = net
        self.file_dir = file_dir
        self.__check_directory()
        self.file_name = file_name
        self.file_path = None
        self.run_id = run_id

    def __check_directory(self):
        if not os.path.isdir(self.file_dir):
            os.makedirs(self.file_dir)

    def __experiment_prefix(self):
        return self.net + "-" + str(self.run_id) if self.run_id != 0 else self.net

    def __construct_results_dict(self, epoch, duration, step, loss, accuracy):
        results_dict = dict()
        results_dict["project_name"] = [self.__experiment_prefix()]
        results_dict["epoch"] = [epoch]
        results_dict["duration(s)"] = [duration]
        results_dict["step(ms)"] = [step]
        results_dict["loss"] = [loss]
        results_dict["accuracy"] = [accuracy]

        return results_dict

    def __find_file(self, mode):
        tmp_list = self.file_name.split(".")
        tmp_list[0] = tmp_list[0] + "_" + mode
        results_file = ".".join(map(str, tmp_list))
        if "train" in mode or "test" in mode:
            return results_file
        else:
            raise ValueError(
                "A wrong mode type was given. Give either 'train' or 'test'."
            )

    def __write_to_csv(self, mode, epoch, duration, step, loss, accuracy):
        results_dict = self.__construct_results_dict(
            epoch, duration, step, loss, accuracy
        )

        results_file = self.__find_file(mode)
        self.file_path = self.file_dir + "/" + results_file

        if not os.path.isfile(self.file_path):
            pd.DataFrame(results_dict).to_csv(self.file_path, index=False)
        else:
            pd.DataFrame(results_dict).to_csv(
                self.file_path, mode="a", header=False, index=False
            )

    def save_results(self, mode, epoch, duration, step, loss, accuracy):
        self.__write_to_csv(mode, epoch, duration, step, loss, accuracy)
