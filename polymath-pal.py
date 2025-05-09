import streamlit as st
import os
import json
import time
from typing import Dict, List, Tuple, Optional
import google.generativeai as genai
# from dotenv import load_dotenv # Not needed for Streamlit if using secrets or direct input
import markdown as md_parser # Renamed to avoid conflict with st.markdown
import sys
import re
from io import StringIO, BytesIO
from datetime import datetime

# --- Configuration & Constants ---
# Load environment variables (Streamlit handles secrets differently)
# load_dotenv() # For local dev, you might still use it, but for deployment, use st.secrets

# Default config (can be overridden by user in UI)
DEFAULT_CONFIG = {
    "gemini_model_name": "gemini-1.5-flash-latest", # Updated to a common recent model
    "temperature": 0.7,
    "max_output_tokens": 8192,
    "system_prompt": "You are a world class SME in RF, Offensive Security, AI, and coding but what sets you apart is your love of teaching and talent for sharing knowledge openly."
}

# Prompt Templates (kept from original script)
COURSE_OBJECTIVE_TEMPLATE = "By the end of this course, learners will be able to {capstone_objective}."
MODULE_TEMPLATE = """
- Module Title: {title}
- Module Objective: {objective}
- Subtopics:
{subtopics}
- Suggested Resources: {resources}
- Small Project: {project}
"""
COURSE_OUTLINE_PROMPT_TEMPLATE = """
Create a comprehensive 8-module course outline based on the following description: {topic_description}. The course should teach learners how to build a {topic_name} from scratch, culminating in a capstone project where they {capstone_objective}. The outline should include:

- An overall course objective: {course_objective}

- For each module:
  - Module title
  - Module objective: What the learner will be able to do after completing this module.
  - List of essential subtopics to be covered (as bullet points).
  - Suggested resources or prerequisites for the module.
  - A small project or exercise at the end of the module that reinforces the learning and contributes to the capstone project.

- Ensure that the modules are logically sequenced, with each building on the previous ones, and that together they cover all necessary skills and knowledge to complete the capstone project.

- Where applicable, include case studies or real-world examples to illustrate key concepts.

- The course should be structured to allow for progressive learning, with assessments or exercises that draw from previous modules where appropriate.

- The course is aimed at learners with basic knowledge in related fields.
"""

# --- Helper Functions (adapted for Streamlit) ---

def initialize_gemini_client(api_key: str, model_name: str) -> Optional[genai.GenerativeModel]:
    if not api_key:
        st.error("GEMINI_API_KEY not provided. Please enter it in the sidebar.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None

def generate_content_with_gemini(
    client: genai.GenerativeModel,
    user_prompt: str,
    system_prompt: str,
    temperature: float,
    max_output_tokens: int
) -> str:
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    try:
        with st.spinner("🤖 Gemini is thinking..."):
            response = client.generate_content(
                contents=full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens
                )
            )
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            st.error(f"Request blocked by API: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}")
            return f"Error: Content generation blocked ({response.prompt_feedback.block_reason})."
        if not response.parts:
            st.warning("Warning: Received an empty response from Gemini.")
            return "Error: Received empty response."
        return response.text.strip() if hasattr(response, 'text') else "Error: Unexpected response format."
    except Exception as e:
        st.error(f"Error generating content via Gemini: {e}")
        if "API key not valid" in str(e):
            st.warning("Please check if your GEMINI_API_KEY is correct and active.")
        return f"Error: Could not generate content. Details: {e}"

def generate_topic_and_description_st(
    client: genai.GenerativeModel,
    user_input: str,
    system_prompt: str,
    temperature: float,
    max_output_tokens: int
) -> Tuple[str, str]:
    prompt_template = """Take the following user input describing what they want to do and transform it into a concise topic title and a detailed topic description. The topic title should be short, clear, and engaging, while the topic description should expand on the input with additional context, purpose, and specifics, written in a natural and informative tone. Here’s the user input: '[INSERT USER INPUT HERE]'. Provide your response in the following format:
Topic Title: [Your title here]
Topic Description: [Your description here]"""
    full_user_prompt = prompt_template.replace("[INSERT USER INPUT HERE]", user_input)
    response_text = generate_content_with_gemini(client, full_user_prompt, system_prompt, temperature, max_output_tokens)

    title_match = re.search(r"Topic Title:\s*(.*?)\s*Topic Description:", response_text, re.DOTALL | re.IGNORECASE)
    description_match = re.search(r"Topic Description:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)

    title = title_match.group(1).strip() if title_match else ""
    description = description_match.group(1).strip() if description_match else ""

    if not title or not description:
        st.warning("Could not parse title and description from LLM response. Displaying raw response.")
        st.text_area("Raw LLM Response for Topic/Description", response_text, height=150)
        # Fallback to using the raw response, or parts of it if possible
        if not title and description: title = "Generated Topic (see description)"
        if not description and title: description = "Generated Description (see title)"
        if not title and not description:
            title = "Error generating title"
            description = response_text # Show the full error

    return title, description


def generate_course_outline_st(
    client: genai.GenerativeModel,
    topic_description: str,
    topic_name: str,
    capstone_objective: str,
    system_prompt: str,
    temperature: float,
    max_output_tokens: int
) -> str:
    course_obj_str = COURSE_OBJECTIVE_TEMPLATE.format(capstone_objective=capstone_objective)
    user_prompt = COURSE_OUTLINE_PROMPT_TEMPLATE.format(
        topic_description=topic_description,
        topic_name=topic_name,
        capstone_objective=capstone_objective,
        course_objective=course_obj_str
    )
    return generate_content_with_gemini(client, user_prompt, system_prompt, temperature, max_output_tokens)

def generate_module_materials_st(
    client: genai.GenerativeModel,
    outline_str: str,
    module_number: int,
    system_prompt: str,
    temperature: float,
    max_output_tokens: int
) -> str:
    user_prompt = f"Create the complete hyper detailed, step-by-step deepdive course materials for module {module_number} based on the course outline provided. Make sure to include code examples where applicable and format the output in markdown."
    full_user_prompt = f"Course Outline:\n{outline_str}\n\n{user_prompt}" # System prompt handled by generate_content
    return generate_content_with_gemini(client, full_user_prompt, system_prompt, temperature, max_output_tokens)

def generate_presentation_st(
    client: genai.GenerativeModel,
    outline_str: str, # Kept for context, though module_content is primary
    module_number: int,
    module_content: str,
    system_prompt: str,
    temperature: float,
    max_output_tokens: int
) -> str:
    user_prompt = f"""
    Create a 10-slide presentation for Module {module_number} based on the following course module content. For each slide, provide:
    1.  Slide Title
    2.  Bullet points summarizing the key content from the module
    3.  A narration script that explains the content of the slide in detail
    Format the output in Markdown, with each slide separated by a horizontal rule (---). Ensure the slides cover the entire module content comprehensively.

    Module Content:
    {module_content}
    """
    # Consider if outline_str is needed in the prompt for presentations
    # full_user_prompt = f"Course Outline:\n{outline_str}\n\n{user_prompt}"
    full_user_prompt = user_prompt # System prompt handled by generate_content
    return generate_content_with_gemini(client, full_user_prompt, system_prompt, temperature, max_output_tokens)

def markdown_to_html_st(md_content: str) -> str:
    html = md_parser.markdown(md_content, extensions=['fenced_code', 'codehilite', 'tables'])
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Course Material</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; padding: 15px; background-color: #f9f9f9; }}
            h1, h2, h3, h4, h5, h6 {{ color: #333; margin-top: 1.5em; margin-bottom: 0.5em; }}
            h1 {{ font-size: 2em; border-bottom: 2px solid #eee; padding-bottom: 0.3em; }}
            h2 {{ font-size: 1.75em; border-bottom: 1px solid #eee; padding-bottom: 0.2em; }}
            h3 {{ font-size: 1.5em; }}
            p {{ margin-bottom: 1em; color: #555; }}
            ul, ol {{ margin-bottom: 1em; padding-left: 20px; }}
            li {{ margin-bottom: 0.5em; }}
            code {{ font-family: 'Courier New', monospace; background-color: #eef; padding: 2px 5px; border-radius: 4px; color: #d03060; }}
            pre {{ background-color: #333; color: #f0f0f0; padding: 15px; overflow-x: auto; border-radius: 5px; border: 1px solid #444; font-size: 0.9em;}}
            pre code {{ background-color: transparent; color: inherit; padding: 0; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            blockquote {{ border-left: 4px solid #ccc; padding-left: 10px; margin-left: 0; color: #777; font-style: italic; }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """

def combine_markdown_content(contents: List[Tuple[str, str]], main_title: str) -> str:
    """Combines multiple markdown contents. List of (title, content)."""
    combined = f"# {main_title}\n\n"
    for i, (title, content) in enumerate(contents, 1):
        combined += f"## Part {i}: {title}\n\n{content}\n\n---\n\n"
    return combined

def parse_module_numbers_st(choice: str, max_modules: int = 8) -> List[int]:
    if not choice: return []
    if choice.lower() == "all":
        return list(range(1, max_modules + 1))
    modules = set() # Use set to avoid duplicates
    for part in choice.replace(" ", "").split(","):
        if not part: continue
        try:
            if "-" in part:
                start_end = part.split("-")
                if len(start_end) == 2:
                    start, end = map(int, start_end)
                    if start <= end:
                        modules.update(range(start, end + 1))
                    else:
                        st.warning(f"Invalid range '{part}': start > end. Skipping.")
                else:
                    st.warning(f"Invalid range format '{part}'. Skipping.")
            else:
                modules.add(int(part))
        except ValueError:
            st.warning(f"Invalid module number '{part}'. Skipping.")
    
    valid_modules = [m for m in sorted(list(modules)) if 1 <= m <= max_modules]
    if len(valid_modules) != len(modules):
        st.warning(f"Some module numbers were outside the valid range (1-{max_modules}) and were excluded.")
    return valid_modules


# --- Streamlit App UI ---
st.set_page_config(page_title="Polymath Pal", layout="wide", initial_sidebar_state="expanded")
st.title("🎓 HVCK's Polymath Pal - Learn all the things")
st.markdown("Powered by Google Gemini. This tool helps you generate comprehensive course structures, materials, and presentations on any topic. This in not built to replace qualified in-person training but to open a wonderful world of learning to any who seek it")

# --- Session State Initialization ---
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
if "gemini_model_name" not in st.session_state:
    st.session_state.gemini_model_name = DEFAULT_CONFIG["gemini_model_name"]
if "temperature" not in st.session_state:
    st.session_state.temperature = DEFAULT_CONFIG["temperature"]
if "max_output_tokens" not in st.session_state:
    st.session_state.max_output_tokens = DEFAULT_CONFIG["max_output_tokens"]
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_CONFIG["system_prompt"]

if "topic_name" not in st.session_state: st.session_state.topic_name = ""
if "topic_description" not in st.session_state: st.session_state.topic_description = ""
if "capstone_objective" not in st.session_state: st.session_state.capstone_objective = "create a functional version of the topic"
if "course_outline" not in st.session_state: st.session_state.course_outline = ""
if "module_materials" not in st.session_state: st.session_state.module_materials = {} # {module_num: content}
if "presentations" not in st.session_state: st.session_state.presentations = {} # {module_num: content}
if "active_stage" not in st.session_state: st.session_state.active_stage = "topic_definition"


# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.session_state.gemini_api_key = st.text_input("Gemini API Key", value=st.session_state.gemini_api_key, type="password", help="Get your API key from Google AI Studio.")
    
    st.session_state.gemini_model_name = st.selectbox(
        "Gemini Model",
        options=["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.0-pro", "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash-preview-04-17"], # Add more as needed
        index=["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.0-pro", "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash-preview-04-17"].index(st.session_state.gemini_model_name)
    )
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05, help="Controls randomness. Lower is more deterministic.")
    st.session_state.max_output_tokens = st.number_input("Max Output Tokens", min_value=512, max_value=8192, value=st.session_state.max_output_tokens, step=256)
    
    with st.expander("Advanced: System Prompt"):
        st.session_state.system_prompt = st.text_area("System Prompt", value=st.session_state.system_prompt, height=150)

    if st.button("Reset to Defaults & Clear Outputs"):
        st.session_state.gemini_model_name = DEFAULT_CONFIG["gemini_model_name"]
        st.session_state.temperature = DEFAULT_CONFIG["temperature"]
        st.session_state.max_output_tokens = DEFAULT_CONFIG["max_output_tokens"]
        st.session_state.system_prompt = DEFAULT_CONFIG["system_prompt"]
        st.session_state.topic_name = ""
        st.session_state.topic_description = ""
        st.session_state.capstone_objective = "create a functional version of the topic"
        st.session_state.course_outline = ""
        st.session_state.module_materials = {}
        st.session_state.presentations = {}
        st.session_state.active_stage = "topic_definition"
        st.success("Settings reset and outputs cleared.")
        st.rerun()

# Initialize Gemini Client (done once if API key changes or at start)
gemini_client = None
if st.session_state.gemini_api_key:
    gemini_client = initialize_gemini_client(st.session_state.gemini_api_key, st.session_state.gemini_model_name)
else:
    st.warning("Please enter your Gemini API Key in the sidebar to enable generation.")

# --- Main Application Flow ---

# Stage 1: Topic Definition
st.header("1. Define Your Course Topic")
topic_source = st.radio(
    "How do you want to define the topic?",
    ["Enter details directly", "Generate from an idea"],
    key="topic_source_radio"
)

if topic_source == "Generate from an idea":
    user_idea = st.text_area("Describe your course idea:", placeholder="e.g., I want to teach people how to build a real-time chat application using Python and WebSockets.", height=100)
    if st.button("💡 Generate Topic & Description from Idea", disabled=not gemini_client or not user_idea):
        if gemini_client and user_idea:
            st.session_state.topic_name, st.session_state.topic_description = generate_topic_and_description_st(
                gemini_client, user_idea, st.session_state.system_prompt, st.session_state.temperature, st.session_state.max_output_tokens
            )
            if st.session_state.topic_name and st.session_state.topic_description and not st.session_state.topic_name.startswith("Error"):
                 st.success("Topic and Description generated!")
            else:
                 st.error("Failed to generate topic and description properly.")


st.session_state.topic_name = st.text_input("Course Topic Name", value=st.session_state.topic_name, placeholder="e.g., Building a Twitter Clone with Django")
st.session_state.topic_description = st.text_area("Detailed Topic Description", value=st.session_state.topic_description, placeholder="A comprehensive course covering...", height=150)
st.session_state.capstone_objective = st.text_input("Capstone Project Objective", value=st.session_state.capstone_objective, placeholder="e.g., build a fully functional microblogging platform")

if st.button("📝 Generate Course Outline", disabled=not gemini_client or not st.session_state.topic_name or not st.session_state.topic_description):
    if gemini_client and st.session_state.topic_name and st.session_state.topic_description:
        st.session_state.course_outline = generate_course_outline_st(
            gemini_client,
            st.session_state.topic_description,
            st.session_state.topic_name,
            st.session_state.capstone_objective,
            st.session_state.system_prompt,
            st.session_state.temperature,
            st.session_state.max_output_tokens
        )
        if st.session_state.course_outline and not st.session_state.course_outline.startswith("Error"):
            st.session_state.active_stage = "outline_display"
            st.success("Course Outline Generated!")
            # Clear previous module/presentation data if outline is regenerated
            st.session_state.module_materials = {}
            st.session_state.presentations = {}
        else:
            st.error("Failed to generate course outline.")
    st.rerun() # Rerun to update UI based on new stage or display outline

# Stage 2: Display Outline and Further Actions
if st.session_state.active_stage == "outline_display" and st.session_state.course_outline:
    st.markdown("---")
    st.header("📜 Course Outline")
    st.markdown(st.session_state.course_outline)

    # Download options for outline
    st.markdown("#### Download Outline:")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            label="⬇️ Download as Markdown (.md)",
            data=st.session_state.course_outline,
            file_name=f"{st.session_state.topic_name.replace(' ', '_')}_outline.md",
            mime="text/markdown",
        )
    with col_dl2:
        html_outline = markdown_to_html_st(st.session_state.course_outline)
        st.download_button(
            label="⬇️ Download as HTML (.html)",
            data=html_outline,
            file_name=f"{st.session_state.topic_name.replace(' ', '_')}_outline.html",
            mime="text/html",
        )
    # JSON for outline is less direct, could be a structured version later
    # For now, just the raw markdown string in a JSON
    with col_dl3:
         outline_json = json.dumps({"course_title": st.session_state.topic_name, "outline_markdown": st.session_state.course_outline}, indent=2)
         st.download_button(
            label="⬇️ Download as JSON (.json)",
            data=outline_json,
            file_name=f"{st.session_state.topic_name.replace(' ', '_')}_outline.json",
            mime="application/json",
        )


    st.markdown("---")
    st.header("📚 Generate Module Content & Presentations")
    
    # Assume 8 modules from prompt, but could be dynamic later
    num_modules_in_outline = 8 # Could try to parse this from outline if it's reliably formatted
    
    module_numbers_text = st.text_input(
        f"Modules to generate (1-{num_modules_in_outline}, e.g., '1-3, 5' or 'all')",
        placeholder="e.g., 1, 3-5, all"
    )
    
    gen_materials = st.checkbox("Generate Module Materials", value=True)
    gen_presentations = st.checkbox("Generate Presentations", value=False)

    if st.button("🚀 Generate Selected Content", disabled=not gemini_client or not module_numbers_text):
        modules_to_gen = parse_module_numbers_st(module_numbers_text, num_modules_in_outline)
        if not modules_to_gen:
            st.warning("Please enter valid module numbers to generate.")
        else:
            total_tasks = len(modules_to_gen) * (gen_materials + gen_presentations)
            progress_bar = st.progress(0)
            tasks_done = 0

            for mod_num in modules_to_gen:
                if gen_materials:
                    st.info(f"Generating materials for Module {mod_num}...")
                    materials = generate_module_materials_st(
                        gemini_client, st.session_state.course_outline, mod_num,
                        st.session_state.system_prompt, st.session_state.temperature, st.session_state.max_output_tokens
                    )
                    if materials and not materials.startswith("Error"):
                        st.session_state.module_materials[mod_num] = materials
                        st.success(f"Materials for Module {mod_num} generated.")
                    else:
                        st.error(f"Failed to generate materials for Module {mod_num}.")
                    tasks_done += 1
                    progress_bar.progress(tasks_done / total_tasks if total_tasks > 0 else 0)

                if gen_presentations:
                    # Need module content to generate presentation
                    module_content_for_pres = st.session_state.module_materials.get(mod_num)
                    if not module_content_for_pres and gen_materials:
                        st.warning(f"Materials for Module {mod_num} were not generated or failed; cannot generate presentation. Skipping.")
                    elif not module_content_for_pres and not gen_materials:
                        st.warning(f"To generate presentations, please also select 'Generate Module Materials' or ensure they were generated previously for Module {mod_num}. Skipping.")
                    else:
                        st.info(f"Generating presentation for Module {mod_num}...")
                        presentation = generate_presentation_st(
                            gemini_client, st.session_state.course_outline, mod_num, module_content_for_pres,
                            st.session_state.system_prompt, st.session_state.temperature, st.session_state.max_output_tokens
                        )
                        if presentation and not presentation.startswith("Error"):
                            st.session_state.presentations[mod_num] = presentation
                            st.success(f"Presentation for Module {mod_num} generated.")
                        else:
                            st.error(f"Failed to generate presentation for Module {mod_num}.")
                    tasks_done += 1
                    progress_bar.progress(tasks_done / total_tasks if total_tasks > 0 else 0)
            
            if total_tasks > 0 : progress_bar.progress(1.0)
            st.session_state.active_stage = "content_display" # Move to display stage
            st.rerun()


# Stage 3: Display Generated Content
if st.session_state.active_stage == "content_display":
    if not st.session_state.module_materials and not st.session_state.presentations:
        st.info("No module content or presentations have been generated yet for the current outline.")
    else:
        st.markdown("---")
        st.header("🎁 Generated Content")

        all_generated_module_keys = sorted(list(set(st.session_state.module_materials.keys()) | set(st.session_state.presentations.keys())))

        if not all_generated_module_keys:
            st.info("No content available for display.")
        else:
            for mod_num in all_generated_module_keys:
                with st.expander(f"Module {mod_num} Content", expanded=False):
                    if mod_num in st.session_state.module_materials:
                        st.subheader(f"Module {mod_num} Materials")
                        st.markdown(st.session_state.module_materials[mod_num])
                        m_col1, m_col2, m_col3 = st.columns(3)
                        with m_col1:
                             st.download_button(f"⬇️ Module {mod_num} MD", st.session_state.module_materials[mod_num], f"module_{mod_num}_materials.md", "text/markdown", key=f"dl_md_mod{mod_num}")
                        with m_col2:
                             st.download_button(f"⬇️ Module {mod_num} HTML", markdown_to_html_st(st.session_state.module_materials[mod_num]), f"module_{mod_num}_materials.html", "text/html", key=f"dl_html_mod{mod_num}")
                        # JSON for module materials would be markdown string in JSON
                        with m_col3:
                            mod_mat_json = json.dumps({f"module_{mod_num}_materials": st.session_state.module_materials[mod_num]}, indent=2)
                            st.download_button(f"⬇️ Module {mod_num} JSON", mod_mat_json, f"module_{mod_num}_materials.json", "application/json", key=f"dl_json_mod{mod_num}")


                    if mod_num in st.session_state.presentations:
                        st.subheader(f"Module {mod_num} Presentation Slides")
                        st.markdown(st.session_state.presentations[mod_num])
                        p_col1, p_col2, p_col3 = st.columns(3)
                        with p_col1:
                            st.download_button(f"⬇️ Pres. {mod_num} MD", st.session_state.presentations[mod_num], f"module_{mod_num}_presentation.md", "text/markdown", key=f"dl_md_pres{mod_num}")
                        with p_col2:
                            st.download_button(f"⬇️ Pres. {mod_num} HTML", markdown_to_html_st(st.session_state.presentations[mod_num]), f"module_{mod_num}_presentation.html", "text/html", key=f"dl_html_pres{mod_num}")
                        with p_col3:
                            pres_json = json.dumps({f"module_{mod_num}_presentation": st.session_state.presentations[mod_num]}, indent=2)
                            st.download_button(f"⬇️ Pres. {mod_num} JSON", pres_json, f"module_{mod_num}_presentation.json", "application/json", key=f"dl_json_pres{mod_num}")


        # Combine files options
        st.markdown("---")
        st.subheader("Combine Generated Content")
        if st.session_state.module_materials:
            if st.button("Combine All Module Materials into One File"):
                combined_mats_list = [(f"Module {k} Materials", v) for k, v in sorted(st.session_state.module_materials.items())]
                combined_mats_md = combine_markdown_content(combined_mats_list, f"{st.session_state.topic_name} - Combined Modules")
                st.session_state.combined_materials_md = combined_mats_md # Store for display/download
            
            if "combined_materials_md" in st.session_state and st.session_state.combined_materials_md:
                 with st.expander("Preview Combined Module Materials", expanded=False):
                      st.markdown(st.session_state.combined_materials_md[:2000] + "...") # Preview first 2000 chars
                 st.download_button("⬇️ Download Combined Materials (MD)", st.session_state.combined_materials_md, f"{st.session_state.topic_name}_combined_modules.md", "text/markdown")
                 st.download_button("⬇️ Download Combined Materials (HTML)", markdown_to_html_st(st.session_state.combined_materials_md), f"{st.session_state.topic_name}_combined_modules.html", "text/html")


        if st.session_state.presentations:
            if st.button("Combine All Presentations into One File"):
                combined_pres_list = [(f"Module {k} Presentation", v) for k, v in sorted(st.session_state.presentations.items())]
                combined_pres_md = combine_markdown_content(combined_pres_list, f"{st.session_state.topic_name} - Combined Presentations")
                st.session_state.combined_presentations_md = combined_pres_md # Store for display/download
            
            if "combined_presentations_md" in st.session_state and st.session_state.combined_presentations_md:
                 with st.expander("Preview Combined Presentations", expanded=False):
                      st.markdown(st.session_state.combined_presentations_md[:2000] + "...") # Preview
                 st.download_button("⬇️ Download Combined Presentations (MD)", st.session_state.combined_presentations_md, f"{st.session_state.topic_name}_combined_presentations.md", "text/markdown")
                 st.download_button("⬇️ Download Combined Presentations (HTML)", markdown_to_html_st(st.session_state.combined_presentations_md), f"{st.session_state.topic_name}_combined_presentations.html", "text/html")

st.markdown("---")
st.caption(f"AI Course Creator v0.2 - Using Gemini model: {st.session_state.gemini_model_name} - Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")