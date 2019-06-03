from timing import RepeatedTimer


class Scheduler:
    """
    this class creates a non-blocking scheduler using the repeated timer
    """
    def __init__(self, function, intervals):
        """

        :param
        function    :   list of functions
        intervals   :   list of intervals
        """
        self.function = function
        self.intervals = intervals

    def run(self):
        for function, interval in zip(self.function, self.intervals):
            rt = RepeatedTimer(interval, function)
