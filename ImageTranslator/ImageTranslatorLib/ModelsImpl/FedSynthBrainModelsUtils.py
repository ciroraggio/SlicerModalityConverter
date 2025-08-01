from monai.transforms import Transform

class CustomResize(Transform):
    def __init__(self, maxSize):
        self.maxSize = maxSize

    def __call__(self, img):
        from monai.transforms import Resize
        
        # Assuming channel-first format
        currentSize = img.shape[1:]
        newSize = [min(dim, self.maxSize) for dim in currentSize]

        if newSize != list(currentSize):
            resize = Resize(spatial_size=newSize)
            return resize(img)
        return img


class PadToCube(Transform):
    def __init__(self, backgroundValue, size=256, mode="constant"):
        self.backgroundValue = backgroundValue
        self.size = size
        self.mode = mode

    def __call__(self, img):
        from torch.nn.functional import pad
        _, d, h, w = img.shape

        padD = (self.size - d) // 2
        padH = (self.size - h) // 2
        padW = (self.size - w) // 2

        padDEnd = self.size - d - padD
        padHEnd = self.size - h - padH
        padWEnd = self.size - w - padW

        padding = (padW, padWEnd, padH, padHEnd, padD, padDEnd)
        paddedImg = pad(input=img, pad=padding, mode=self.mode, value=self.backgroundValue)

        return paddedImg


class MRINormalize(Transform):
    def __init__(self, type="minmax"):
        self.type = type

    def __call__(self, img):
        from torch import nan_to_num
        if self.type == "minmax":
            img -= img.min()
            img /= img.max() - img.min()
            img = nan_to_num(img, nan=0)
            return img