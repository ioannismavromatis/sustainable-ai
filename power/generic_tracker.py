import os
import ssl

import utils.log as logger

LOGGER = os.environ.get("LOGGER", "info")

custom_logger = logger.get_logger(__name__)
custom_logger = logger.set_level(__name__, LOGGER)
custom_logger.debug("Logger initiated: %s", custom_logger)

ssl._create_default_https_context = ssl._create_unverified_context
VALID_TOOLS = ["eco2ai", "codecarbon", "carbontracker"]


class GenericTracker:
    def __init__(self, args, net):
        self.args = args
        self.net = net.__class__.__name__
        self.standardised_tool = None
        self.power_tool = self.power_evaluation()

    def import_module(self):
        if self.args.evaluation_tool == "eco2ai":
            power_tool = __import__(self.args.evaluation_tool)
            power_tool.set_params(
                project_name="eco2ai",
                experiment_description=self.net,
                file_name="./results/resultsEco2AI.csv",
            )
        elif self.args.evaluation_tool == "carbontracker":
            power_tool = __import__(self.args.evaluation_tool + ".tracker")

        return power_tool

    def power_evaluation(self):
        if self.args.evaluation_tool is not None:
            tool = self.args.evaluation_tool.lower()
            self.standardised_tool = tool
            if self.args.evaluation_tool not in VALID_TOOLS:
                raise ValueError(
                    f'Tool "{self.args.evaluation_tool}" is not available tool for evaluation. Available tools are "Eco2AI", "CodeCarbon", and "CarbonTracker"'
                )

            custom_logger.info(
                f"Tool '{self.args.evaluation_tool}' will be used for power performance evaluation"
            )

            power_tool = self.import_module()

            if tool == "eco2ai":
                tracker = power_tool.Tracker(cpu_processes="all", ignore_warnings=True)
            elif tool == "carbontracker":
                tracker = power_tool.tracker.CarbonTracker(
                    epochs=self.args.epochs,
                    monitor_epochs=self.args.epochs,
                    log_dir="./results/",
                    log_file_prefix=self.net,
                )

            return tracker
        else:
            custom_logger.info(
                "No tool will be used for the power performance evaluation"
            )

            return None

    def start(self):
        if self.standardised_tool == "eco2ai":
            self.power_tool.start()
        elif self.standardised_tool == "carbontracker":
            self.power_tool.epoch_start()

    def stop(self):
        if self.standardised_tool == "eco2ai":
            self.power_tool.stop()
        elif self.standardised_tool == "carbontracker":
            self.power_tool.epoch_end()
