class BasicDescriptor(object):
    def __init__(self, inplace=True, batched=True):
        self.inplace = inplace
        self.batched = batched

    def __call__(self, data):
        # data["num_nodes"] = data["x"].shape[0]
        return data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
