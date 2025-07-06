SYSTEM_PORMPT = """You are an expert vision assistant. Your task is to analyze images and provide detailed descriptions, insights, and answers based on the visual content.
You should focus on the objects, actions, text and context present in the images.
Format your answer exactly as requested by the user, do not add any extra words or sentences. Ensure that your responses are concise and relevant to the visual content."""

USER_PROMPT = """
Analyze this image and provide a detailed description exactly in this format:

            SUBJECT: What is the main theme or subject of this photo?
            PERSONS: Are there any people in the photo? If yes, describe them (number, age group, activities, etc.)
            TEXT: Include all and any text that is visible in the photo. Include signs, labels, writing, etc.
            CATEGORIES: What categories does this image belong to? (e.g., landscape, portrait, 3d render, photorealistic, fantasy, etc.)
            OBJECTS: List the main objects in the image and their positions.
            DETAILED DESCRIPTION: Provide a comprehensive description of the image including:
            - Setting/location
            - Objects and their positions
            - Colors and lighting
            - Mood/atmosphere
            - Any notable details
            """

MAX_IMAGE_DIMENSION = 2048

# VISION_MODEL_NAME = "mistral-small3.1:latest"
VISION_MODEL_NAME = "gemma3:12b"
# VISION_MODEL_NAME = "llama3.2-vision:latest"
EMBEDDINGS_MODEL_NAME = "nomic-embed-text:latest"