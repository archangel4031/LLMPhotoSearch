# app.py
from pathlib import Path
from modules.config import *
from modules.chroma_vector_store import ChromaVectorStore
from modules.process_image import ImageProcessor
import gradio as gr
import tkinter as tk
from tkinter import filedialog

vector_store = ChromaVectorStore(embeddings_model=EMBEDDINGS_MODEL_NAME)
current_directory = Path.cwd()


def select_folder():
    """Opens a dialog to select a folder and returns the path."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)  # Make dialog appear on top
    folder_path = filedialog.askdirectory(title="Select Image Folder")
    root.destroy()
    if folder_path:
        return folder_path
    return gr.update()  # Do not update the textbox if no folder is selected


# Placeholder functions - replace with your actual implementations
def process_images(folder_path):
    """
    Processes new images in a folder, skipping those already in the vector store.
    Yields progress updates for the Gradio interface.
    """
    image_processor = ImageProcessor(model_name=VISION_MODEL_NAME)

    # Check if folder exists
    folder = Path(folder_path)
    if not folder.is_dir():
        yield f"Error: Folder does not exist or is not a directory: {folder_path}"
        return

    # Get all supported image files from the folder
    all_image_files = [
        f
        for f in folder.rglob("*")
        if f.is_file() and image_processor._is_image_file(f)
    ]
    total_images_in_folder = len(all_image_files)

    if total_images_in_folder == 0:
        yield "No images found in the specified folder."
        return

    # Get filenames already in the vector store to avoid re-processing
    existing_filenames = vector_store.get_existing_filenames()

    # Filter out images that are already processed
    new_image_files = [f for f in all_image_files if f.name not in existing_filenames]

    num_skipped = total_images_in_folder - len(new_image_files)
    total_to_process = len(new_image_files)

    results_output = f"Found {total_images_in_folder} images in folder.\n"
    if num_skipped > 0:
        results_output += f"Skipping {num_skipped} image(s) already in the vector store.\n"

    if total_to_process == 0:
        results_output += "\n✅ No new images to process."
        yield results_output
        return

    results_output += f"Processing {total_to_process} new image(s)...\n\n"

    # Process each new image and build results
    processed_images = []
    for i, image_file in enumerate(new_image_files):
        # Process the image
        result = image_processor.process_single_image(str(image_file))
        if result:
            processed_images.append(result)
            results_output += f"### {image_file.name}\n{result.get('description', 'No description available.')}\n\n**Progress: {i+1}/{total_to_process}**\n\n"
        yield results_output  # This will update the output in real-time

    # Update vector store with all newly processed images
    if processed_images:
        vector_store.update_vector_store(processed_images)
        results_output += f"\n✅ Successfully processed {len(processed_images)}/{total_to_process} new images and updated the vector store."
    else:
        results_output += "\n❌ No new images were processed successfully."

    yield results_output


def search_images(query, k_value):
    # Your search logic here
    # Load existing vector store
    vector_store.load_vector_store()

    print(f"Current working directory: {current_directory}")

    # Search for photos
    results = vector_store.search_photos(query, k=k_value)

    image_paths = []
    if results:
        print(f"\nFound {len(results)} results for query: '{query}'")
        print("=" * 50)

        # Create Path for images
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"File: {result['filename']}")
            print(f"Path before conversion: {result['image_path']}")
            # The path from the DB might start with a separator (e.g., '\content\photos...'),
            # which causes pathlib to treat it as an absolute path from the drive root.
            # We strip leading separators to ensure it's treated as a relative path.
            relative_path_str = str(result['image_path']).lstrip('/\\')
            result['image_path'] = current_directory / relative_path_str
            print(f"Path after conversion: {result['image_path']}")
            print(f"Similarity Score: {result['similarity_score']:.4f}")
            print(f"Description: {result['content'][:200]}...")
            print("-" * 40)
            image_paths.append(result["image_path"])
    else:
        print(f"No results found for query: '{query}'")
    # Return both image paths for the gallery and the full results for state
    return image_paths, results


def show_image_details(results_data, evt: gr.SelectData):
    """
    Displays the details of the selected image in the gallery.
    This function is triggered by the 'select' event on the gallery.
    """
    if not results_data:
        return "Click an image in the gallery to see its details here."

    # Get the details of the selected image using the index from the event
    selected_image = results_data[evt.index]

    # Format the details into a Markdown string for display
    filename = selected_image.get("filename", "N/A")
    path = selected_image.get("image_path", "N/A")
    description = selected_image.get("content", "N/A")
    score = selected_image.get("similarity_score", 0)

    # The 'path' variable already contains the full, absolute Path object from the search_images function.
    details_md = f"""### Image Details\n**Filename:** `{filename}`\n\n**Path:** `{path}`\n\n**Similarity Score:** `{score:.4f}`\n\n---\n**Description:**\n{description}"""
    return details_md


# Create the Gradio interface with tabs
with gr.Blocks() as demo:
    gr.Markdown("# LLM Photo Search")
    gr.Markdown(
        """
        **Instructions:**

        1.  **Process Images**: Enter the path to a folder containing images you want to process. Click the "Process Images" button to analyze and store them in the vector database.
        2.  **Search Images**: Enter a search query and adjust the number of results (k) using the slider. Click the "Search" button to find images that match your query.
        """
    )
    with gr.Tabs():
        with gr.Tab("Process Images"):
            with gr.Row():
                folder_input = gr.Textbox(
                    label="Folder Path",
                    placeholder="Enter the path to your image folder",
                    scale=8,
                )
                folder_button = gr.Button("Browse", variant="secondary", scale=1)

            with gr.Row():
                process_button = gr.Button("Process Images", variant="primary")
            with gr.Row():
                # Output area for processing results
                output_process = gr.Markdown(value = "Processing Results", label="Processing Results", container=True, max_height=500)

                process_button.click(
                    fn=process_images,
                    inputs=folder_input,
                    outputs=output_process,
                    show_progress=True,  # Shows Gradio's built-in progress indicator
                )

                folder_button.click(fn=select_folder, inputs=None, outputs=folder_input)

        with gr.Tab("Search Images"):
            # To store the full search results data, which includes metadata
            results_state = gr.State([])

            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        search_input = gr.Textbox(
                            label="Search Query",
                            placeholder="Enter your search query",
                            scale=4,
                        )
                        k_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Number of Results (k)",
                            scale=1,
                        )
                    search_button = gr.Button("Search", variant="primary")
                    output_gallery = gr.Gallery(
                        label="Search Results",
                        show_label=False,
                        elem_id="gallery",
                        columns=4,
                        height="auto",
                    )
                with gr.Column(scale=2):
                    image_details_display = gr.Markdown(
                        label="Image Details",
                        value="Click an image in the gallery to see its details here.",
                    )

            search_button.click(
                fn=search_images,
                inputs=[search_input, k_slider],
                outputs=[output_gallery, results_state],
            )

            output_gallery.select(
                fn=show_image_details,
                inputs=[results_state],
                outputs=image_details_display,
            )

# Launch the app
if __name__ == "__main__":
    demo.launch()
