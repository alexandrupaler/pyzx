"""
Measurement module
"""
import time


class MeasurementDict(dict):
    """
    Wrapper class for timing dictionary
    """

    def __repr__(self):
        return f"Default dict: {self}"

    def __str__(self):
        result = ""
        for method_name, running_time in self.items():
            result += f"{method_name} - {running_time} \n"

        return result


timing = MeasurementDict()


def timeit(method):
    """
    Decorator for measuring the execution time of a given method
    :param method: method for which to run a time measurement
    :return: decorated method
    """

    def timed(*args, **kw):
        owner_id = getattr(args[0], "name") if args else None

        time_start = time.time()
        result = method(*args, **kw)
        time_end = time.time()

        running_time = int((time_end - time_start) * 1000)

        if owner_id:
            pipeline_timing = getattr(args[0], "timing")
            pipeline_timing[method.__name__] = running_time
        else:
            timing[method.__name__] = running_time
        # print(f"{prefix}{method.__name__} {int((time_end - time_start) * 1000) }ms")

        return result

    return timed
