from dataset import Data


class Compose(object):
    """
    A class to compose multiple transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data: Data) -> Data:
        """
        Args: data (Data): input data
        Returns: data (Data): transformed data
        """
        for t in self.transforms:
            # go through each transform and apply it to the data
            data = t(data)
        return data
