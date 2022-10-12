import os
import pathlib
from tkinter import Image
import typing
import torchvision
from torchvision.datasets.imagenet import parse_devkit_archive, parse_train_archive, parse_val_archive

class ImageNet(torchvision.datasets.ImageNet):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

Args:
    root (string): Root directory of the ImageNet Dataset.
    split (string, optional): The dataset split, supports ``train``, or ``val``.
    transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    loader (callable, optional): A function to load an image given its path.

 Attributes:
    classes (list): List of the class name tuples.
    class_to_idx (dict): Dict with items (class_name, class_index).
    wnids (list): List of the WordNet IDs.
    wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
    imgs (list): List of (image path, class_index) tuples
    targets (list): The class_index value for each image in the dataset

Returns:
    tuple: (sample, target) where target is class_index of the target class."""
    def __init__(self, root: str, split: str = ("train", "val"), **kwargs: typing.Any):
      parse_devkit_archive(root)
      parse_train_archive(root)
      parse_val_archive(root)
      super().__init__(root, split, **kwargs)

userHome = os.path.join(pathlib.Path.home(), "TorchStudio")
os.chdir(userHome)
imagenet = ImageNet(os.path.join("data", "imagenet"))