from utils import log

custom_logger = log.get_logger(__name__)
custom_logger = log.set_level(__name__, "info")

VALID_TOOLS = ["eco2ai", "codecarbon", "carbontracker"]


class GenericTracker:
    def __init__(self, args, net):
        self.args = args
        self.net = net.__class__.__name__
        self.standardised_tool = None
        self.power_tool = self.power_evaluation()

    def _experiment_prefix(self):
        return (
            self.net + "-" + str(self.args.run_id)
            if self.args.run_id != 0
            else self.net
        )

    def import_module(self):
        if self.args.tool == "eco2ai":
            power_tool = __import__(self.args.tool)
        elif self.args.tool == "carbontracker":
            power_tool = __import__(self.args.tool + ".tracker")
        elif self.args.tool == "codecarbon":
            power_tool = __import__(self.args.tool)

        return power_tool

    def power_evaluation(self):
        if self.args.tool != "":
            tool = self.args.tool.lower()
            self.standardised_tool = tool
            if self.args.tool not in VALID_TOOLS:
                raise ValueError(
                    f'Tool "{self.args.tool}" is not available tool for evaluation. Available tools are "Eco2AI", "CodeCarbon", and "CarbonTracker"'
                )

            custom_logger.info(
                "Tool '%s' will be used for power performance evaluation",
                self.args.tool,
            )

            power_tool = self.import_module()

            if tool == "eco2ai":
                power_tool.set_params(
                    project_name="eco2ai",
                    experiment_description=self._experiment_prefix(),
                    file_name="./results/resultsEco2AI.csv",
                )
                tracker = power_tool.Tracker(cpu_processes="all", ignore_warnings=True)
            elif tool == "carbontracker":
                tracker = power_tool.tracker.CarbonTracker(
                    epochs=self.args.epochs,
                    monitor_epochs=self.args.epochs,
                    log_dir="./results/",
                    log_file_prefix=self._experiment_prefix(),
                )
            elif tool == "codecarbon":
                tracker = power_tool.EmissionsTracker(
                    save_to_file=True,
                    output_dir="./results/",
                    project_name=self._experiment_prefix(),
                )

            return tracker
        else:
            custom_logger.info(
                "No tool will be used for the power performance evaluation"
            )

            return None

    def start(self):
        if self.standardised_tool == "eco2ai" or self.standardised_tool == "codecarbon":
            self.power_tool.start()
        elif self.standardised_tool == "carbontracker":
            self.power_tool.epoch_start()

    def stop(self):
        if self.standardised_tool == "eco2ai" or self.standardised_tool == "codecarbon":
            results = self.power_tool.stop()
        elif self.standardised_tool == "carbontracker":
            results = self.power_tool.epoch_end()

        return results

    def get_tracker(self):
        if self.power_tool is None:
            return False

        return True
