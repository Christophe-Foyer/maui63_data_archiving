from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from maui63_data_archiving.image_source.abstract import ImageSource


class Converter(ABC):
    """Base class for converters."""
    
    @abstractmethod
    def convert(self, source: ImageSource, output_path: Union[str, Path], **kwargs):
        """Convert images from source to output."""
        pass
