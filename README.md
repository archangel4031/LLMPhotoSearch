
# 🖼️ PhotoTalk: Photo Search with LLM-Powered Descriptions

This project allows users to **search their personal photo collection using natural language**, by leveraging local LLMs to **generate and store private photo descriptions**.

Powered by:
- 🧠 Local LLMs with [Ollama](https://ollama.com/)
- 🔍 Vector search with [Chroma DB](https://www.trychroma.com/)
- 🖥️ Clean interface using [Gradio](https://gradio.app/)

---

## ✨ Features

- 📸 **Private & Local Photo Description Generation**  
  Uses vision-capable LLMs via [Ollama](https://ollama.com/) (e.g., `gemma:3`, `llama3-vision`, `mistral-small`) to generate accurate and detailed photo captions.

- 🧾 **Vector Embedding and Storage**  
  Captions and metadata are embedded and stored in a local [Chroma](https://www.trychroma.com/) vector store.

- 🔍 **Natural Language Search**  
  Users can search for photos using everyday language, like _“Pictures with mountains at sunset”_.

- 🖥️ **User-Friendly Interface**  
  Gradio web UI makes uploading, describing, and searching intuitive and accessible.

- 🔐 **100% Local and Private**  
  All processing and storage happens locally. No photos or data leave your machine.

---

## Gradio Interface

![Gradio UI](https://github.com/archangel4031/LLMPhotoSearch/blob/main/screenshots/Screenshot1.png?raw=true)

![Gradio UI](https://github.com/archangel4031/LLMPhotoSearch/blob/main/screenshots/Screenshot2.png?raw=true)

![Gradio UI](https://github.com/archangel4031/LLMPhotoSearch/blob/main/screenshots/Screenshot3.png?raw=true)

---

## 🚀 Getting Started

### 1. Install Dependencies

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

```bash
# Create a virtual environment using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
uv pip install -r requirements.txt
```

### 2. Start Ollama

Make sure [Ollama](https://ollama.com/) is installed and has required models downloaded:

```bash
ollama pull nomic-embed-text:latest
ollama pull gemma3:12b
```
You can use other LLMs for vision models e.g. Mistral or Llama3.2 if needed.

### 3. Launch the App

```bash
gradio app.py
```

This will launch the Gradio interface in your browser.

---

## 🧠 Model Configuration

You can choose between supported models by setting the `VISION_MODEL_NAME` variable in your modules/config:

```python
VISION_MODEL_NAME = "gemma3:12b"
```

Ensure the model is downloaded in Ollama with:

```bash
ollama pull <model-name>
```

---

## 📁 Folder Structure

```text
.
├── app.py                          # Main application file
└── modules/                        # Modules folder
    ├── process_image.py            # Image handling and metadata
    ├── vision_llm.py               # LLM caption generation
    ├── chroma_vector_store.py      # Chroma DB logic
    ├── config.py                   # Configurations and model paths
├── requirements.txt
└── README.md
```

---

## 🔍 Example Query Prompts

Try natural queries like:

* “Photos from the beach”
* “kids and birthday cake”
* “night city photos with lights”

---

## 📒 Jupyter Notebook on Google Colab

A Google Colab Notebook is provided for those poor souls who do not have a powerful graphics card. The notebook will also use Ollama for image processing and generate a vector store, which can then be used offline.

[Colab for Image Processing](https://colab.research.google.com/drive/1Ev4veQRl2mSIBC9gkA9LfHkBDIHnxdYY?usp=sharing)

---


## 🛡️ Privacy & Security

* No external API calls.
* No cloud storage.
* 100% local vector storage with Chroma.
* Perfect for photographers, researchers, or privacy-focused users.

---

## 📌 Roadmap

* [ ] Add tagging support
* [ ] Filter by metadata
* [ ] Move config settings in UI

---

## 🤝 Contributions

Feel free to fork, open issues, or contribute via pull requests!
This project is designed to be privacy-first and community-friendly.

---


## 🙌 Acknowledgements

* [Ollama](https://ollama.com/)
* [Chroma DB](https://www.trychroma.com/)
* [Gradio](https://www.gradio.app/)
* LLM models by Meta, Google, Mistral, and others.

---
