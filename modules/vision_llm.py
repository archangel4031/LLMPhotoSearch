from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from pathlib import Path
import os
from PIL import Image
import io
import base64
from typing import TypedDict
from modules.config import *

class ImageAnalysisResult(TypedDict):
    image_path: str
    filename: str
    description: str
    file_size: int

def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""

    try:
        # Infer image type from extension for the data URI
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"  # Common alias

        # Check for supported input image formats that Pillow can open and we want to process
        if ext not in ["png", "jpeg", "gif", "webp"]:
            raise ValueError(
                f"Unsupported image extension: {ext}. "
                "Supported formats are PNG, JPEG, GIF, WEBP."
            )

        # Open the image with Pillow
        img = Image.open(image_path)

        # Resize the image while maintaining aspect ratio
        # The thumbnail method resizes the image in place to fit within the given dimensions
        # img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))

        # Save the resized image to an in-memory buffer
        buffer = io.BytesIO()
        # Determine the format for Pillow's save method (Pillow uses 'JPEG' for .jpeg)
        pil_save_format = ext.upper()
        if (
            pil_save_format == "JPG"
        ):  # Should already be 'JPEG' due to earlier normalization
            pil_save_format = "JPEG"
        img.save(buffer, format=pil_save_format)
        image_bytes = buffer.getvalue()

        # Encode the bytes from the buffer to base64
        encoded_string = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/{ext};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        raise
    except Exception as e:
        print(f"Error encoding image {image_path} to base64: {e}")
        raise


def call_vision_llm(
    image_path: str,
    system_prompt: str = SYSTEM_PORMPT,
    prompt: str = USER_PROMPT,
    model_name: str = "gemma3:12b",
) -> ImageAnalysisResult:
    """
    Call a vision LLM with an image and a prompt.

    Args:
        image_path (str): Path to the image file.
        prompt (str): Text prompt to send to the LLM.
        model_name (str): Name of the vision model in Ollama (default: 'llava').
        embeddings_model (str): Name of the embeddings model in Ollama (default: 'llama2').

    Returns:
        dict: Processed image data including description and metadata.
    """
    llm = ChatOllama(model=model_name, temperature=0.1)

    try:
        print(f"Processing: {image_path}")

        # Encode image to base64
        base64_image = encode_image(image_path)

        llm_input = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {"type": "image_url", "image_url": {"url": base64_image}},
                ]
            ),
        ]

        # Generate description using the LLM
        description = llm.invoke(llm_input)
        print("=" * 40)
        print(f"Generated description for {image_path}:\n{description.content}\n")
        print("=" * 40)

        return {
            "image_path": image_path,
            "filename": Path(image_path).name,
            "description": description.content,
            "file_size": os.path.getsize(image_path),
        }

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None
