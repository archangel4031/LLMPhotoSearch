from typing import List, Dict, Any
from pathlib import Path
from modules.config import *
from modules.vision_llm import call_vision_llm
from typing import TypedDict

class ImageAnalysisResult(TypedDict):
    image_path: str
    filename: str
    description: str
    file_size: int

class ImageProcessor:
    """
    A class to handle processing of single images or folders of images.
    """

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    def __init__(self, model_name: str = VISION_MODEL_NAME):
        """
        Initializes the ImageProcessor.

        Args:
            model_name (str): The name of the vision model to use for processing.
        """
        self.model_name = model_name

    def _is_image_file(self, file_path: Path) -> bool:
        """Check if a file is a supported image format."""
        return file_path.suffix.lower() in self.SUPPORTED_FORMATS

    def process_single_image(self, image_path: str) -> ImageAnalysisResult:
        """Process a single image and return structured data."""
        return call_vision_llm(
            image_path=image_path,
            system_prompt=SYSTEM_PORMPT,
            prompt=USER_PROMPT,
            model_name=self.model_name,
        )

    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Process all images in a folder."""
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"Folder does not exist or is not a directory: {folder}")

        processed_images = []

        image_files = [
            f for f in folder.rglob("*") if f.is_file() and self._is_image_file(f)
        ]

        print(f"Found {len(image_files)} image files to process.")

        for image_file in image_files:
            result = self.process_single_image(str(image_file))
            if result:
                processed_images.append(result)

        return processed_images