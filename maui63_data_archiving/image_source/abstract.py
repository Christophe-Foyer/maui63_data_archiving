from abc import ABC, abstractmethod
from typing import List

import numpy as np


class ImageSource(ABC):
    """Base class for image sources."""
    
    @abstractmethod
    def __len__(self) -> List[np.ndarray]:
        """Return list of images as numpy arrays."""
        pass

    @abstractmethod
    def get_image(self, idx) -> np.ndarray:
        """Return the image at idx."""
        pass

    def __getitem__(self, idx):
        return self.get_image(idx)

    def get_images(self) -> List[np.ndarray]:
        return [self.get_image(idx) for idx in range(len(self))]

    @abstractmethod
    def get_image_path(self, idx) -> str:
        """Return image file."""
        pass
