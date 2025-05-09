# 🎓 Polymath Pal: AI Course Creator

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)](YOUR_STREAMLIT_APP_URL_HERE) <!-- Replace with your app URL -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Polymath Pal** is a Streamlit-powered web application designed to help users generate comprehensive course structures, detailed module materials, and presentation outlines on any topic they wish to learn or teach. Powered by Google's Gemini models, this tool aims to democratize learning by making course creation accessible and efficient, particularly for cutting-edge or niche subjects.

The core philosophy is inspired by a "love of teaching and talent for sharing knowledge openly," aiming to assist those with a passion for learning to expand their horizons.

## ✨ Features

*   **AI-Powered Topic Definition:**
    *   Enter course details directly.
    *   Generate a concise topic title and detailed description from a broader course idea using Gemini.
*   **Comprehensive Course Outline Generation:**
    *   Creates an 8-module course outline based on topic name, description, and a capstone project objective.
    *   Includes overall course objective, and for each module: title, objective, subtopics, suggested resources, and a small reinforcing project.
*   **Detailed Module Material Generation:**
    *   Generates hyper-detailed, step-by-step course materials for selected modules, including code examples where applicable.
*   **Presentation Slide Generation:**
    *   Creates a 10-slide presentation outline (title, bullet points, narration script) in Markdown for selected modules.
*   **Flexible Configuration:**
    *   Choose from various Gemini models (e.g., `gemini-1.5-flash-latest`, `gemini-1.5-pro-latest`, etc.).
    *   Adjust AI parameters like temperature and max output tokens.
    *   Customize the system prompt to guide the AI's persona and output style.
*   **Downloadable Content:**
    *   Download course outlines, module materials, and presentations in Markdown (.md), HTML (.html), and JSON (.json) formats.
    *   Option to combine all module materials or presentations into single downloadable files.
*   **User-Friendly Interface:**
    *   Intuitive multi-stage Streamlit application flow.
    *   Clear progress indicators and error handling.
    *   Responsive design for various screen sizes.



## 📋 Prerequisites

Before you begin, ensure you have the following installed:

*   **Python:** Version 3.8 or higher.
*   **Pip:** Python package installer.
*   **Git:** (Optional, for cloning the repository).
*   **Google Gemini API Key:** You will need a valid API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## ⚙️ Installation & Setup

1.  **Clone the Repository (Optional):**
    If you have the code locally, navigate to the project directory. Otherwise:
    ```bash
    git clone https://github.com/scs-labrat/polymath-pal.git
    cd polymath-pal
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    streamlit
    google-generativeai
    Markdown
    python-dotenv  # For local development to load .env files
    ```
    Then install the packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    You need to provide your Gemini API key.

    *   **For Local Development:**
        Create a `.env` file in the root of your project directory and add your API key:
        ```env
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
        ```
        The application uses `python-dotenv` to load this variable if the script were adapted for it. Currently, the script expects the API key to be entered in the Streamlit sidebar or be available as an environment variable that Streamlit can pick up. For direct local run, you might modify the code to use `load_dotenv()` or set the environment variable `GEMINI_API_KEY` in your shell.

    *   **For Streamlit Cloud Deployment:**
        Store your `GEMINI_API_KEY` as a secret in your Streamlit Cloud app settings. The application will automatically pick it up.

## ▶️ How to Run

1.  Ensure your virtual environment is activated.
2.  Make sure your `GEMINI_API_KEY` is accessible (either as an environment variable or you'll enter it in the app).
3.  Run the Streamlit application:
    ```bash
    streamlit run polymath_pal.py
    ```
    (Replace `polymath_pal.py` with the actual name of your Python script if different).

4.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## 🛠️ Usage

Once the application is running:

1.  **Sidebar Configuration:**
    *   Enter your **Gemini API Key** if not already set.
    *   Select the desired **Gemini Model**. Note: The application includes newer preview models; ensure your API key has access to them.
    *   Adjust **Temperature** and **Max Output Tokens** as needed.
    *   (Advanced) Modify the **System Prompt** to change the AI's default behavior.
    *   Click "Reset to Defaults & Clear Outputs" to clear all generated content and settings.

2.  **Stage 1: Define Your Course Topic:**
    *   Choose "Enter details directly" to input your topic name, description, and capstone objective manually.
    *   Or, choose "Generate from an idea," describe your course idea, and click "💡 Generate Topic & Description from Idea" for the AI to suggest these.
    *   Once the topic details are filled, click "📝 Generate Course Outline."

3.  **Stage 2: Review Course Outline & Generate Content:**
    *   If the outline is generated successfully, it will be displayed.
    *   You can download the outline in Markdown, HTML, or JSON format.
    *   Proceed to "📚 Generate Module Content & Presentations."
    *   Enter the module numbers you want to generate content for (e.g., "1", "2-4", "all").
    *   Select whether to generate "Module Materials" and/or "Presentations."
    *   Click "🚀 Generate Selected Content." This may take some time depending on the number of modules and content types selected.

4.  **Stage 3: Display & Download Generated Content:**
    *   After generation, generated materials and presentations will be displayed in expandable sections for each module.
    *   Download options (MD, HTML, JSON) are available for each individual piece of content.
    *   Use the "Combine Generated Content" section to merge all module materials or all presentations into single files for download.

## 🔧 How It Works

Polymath Pal follows a structured, multi-stage process:

1.  **Initialization:** The Streamlit app sets up the UI and initializes session state variables for storing configurations and generated content.
2.  **Gemini Client:** The `initialize_gemini_client` function configures the Google Generative AI client using the provided API key and selected model.
3.  **Content Generation:**
    *   **Topic & Description:** `generate_topic_and_description_st` uses a specific prompt template to instruct Gemini to create a topic title and description from a user's idea.
    *   **Course Outline:** `generate_course_outline_st` uses the `COURSE_OUTLINE_PROMPT_TEMPLATE` to request a comprehensive 8-module outline, ensuring logical flow and inclusion of objectives, subtopics, resources, and projects.
    *   **Module Materials:** `generate_module_materials_st` prompts Gemini to create detailed, step-by-step content for a specific module based on the previously generated outline.
    *   **Presentations:** `generate_presentation_st` prompts Gemini to create a 10-slide presentation outline (titles, bullets, narration) based on the generated module materials.
    All generation functions call `generate_content_with_gemini`, which handles the interaction with the Gemini API, including managing spinners, error handling, and parsing the response.
4.  **Markdown Processing:** The `markdown_to_html_st` function uses the `markdown` library (aliased as `md_parser`) to convert generated Markdown content into styled HTML for preview and download.
5.  **State Management:** Streamlit's session state (`st.session_state`) is used extensively to preserve user inputs, configurations, and generated content across interactions and reruns. This allows for a step-by-step generation process.
6.  **User Interface:** The application is built with Streamlit widgets (`st.text_input`, `st.text_area`, `st.selectbox`, `st.slider`, `st.button`, `st.download_button`, `st.expander`, etc.) to provide an interactive experience.

The system prompt ("You are a world class SME...") plays a crucial role in guiding the AI's tone, depth, and pedagogical style throughout the content generation process.

## 📁 Project Structure (Simplified)

```
polymath-pal/
├── polymath_pal.py         # Main Streamlit application script
├── requirements.txt        # Python dependencies
├── .env                    # (Optional, for local API key storage)
└── README.md               # This file
```

## 💡 Future Enhancements

*   **Fact-Checking Integration:** Option to send generated content to other LLMs (GPT, Claude) for automated fact-checking and suggestions directly within the app.
*   **More Granular Control over Outline:** Allow users to edit/regenerate specific parts of the outline.
*   **Dynamic Number of Modules:** Allow users to specify the desired number of modules.
*   **More Output Formats:** Support for formats like PDF, PowerPoint (.pptx) for presentations.
*   **Interactive Previews:** Richer HTML previews within the app.
*   **Advanced Prompt Engineering Options:** UI for users to more deeply customize prompts for each generation stage.
*   **Saving/Loading Course Projects:** Ability to save the entire state of a course generation project and load it later.
*   **Community Sharing:** (Ambitious) A platform for users to share generated courses or templates.

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements or find bugs, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature` or `bugfix/YourBug`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

Please ensure your code follows good practices and includes comments where necessary.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details (if you create one, otherwise state "MIT License").

## 🙏 Acknowledgements

*   **Google Gemini:** For providing the powerful generative AI capabilities.
*   **Streamlit Team:** For the excellent framework for building web applications in Python.
*   The open-source community for the libraries used in this project.

---

Thank you for your interest in Polymath Pal! May your love of learning flourish.
```
