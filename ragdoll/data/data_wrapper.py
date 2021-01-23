
class DataWrapper:
    def __init__(self, data):
        self.data = data

    def get_attr(self, attr):
        if self.data is None:
            return DataWrapper(None)
        assert hasattr(self.data, attr)
        return DataWrapper(getattr(self.data, attr))

    # func must have a return value
    def call_func(self, func, *args, **kwargs):
        if self.data is None:
            return DataWrapper(None)
        assert hasattr(self.data, func)
        return DataWrapper(getattr(self.data, func)(*args, **kwargs))

    def other_func(self, func, *args, **kwargs):
        if self.data is None:
            return DataWrapper(None)
        ret = func(self.data, *args, **kwargs)
        return DataWrapper(ret)

    def get_val(self, default=None):
        if self.data is None:
            return default
        return self.data
