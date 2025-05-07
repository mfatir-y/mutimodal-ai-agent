import json
import os
import traceback

import streamlit as st

from main import initialize_ai_components
from model_registry import CHAT_MODELS, CODE_MODELS

# Set page configuration
st.set_page_config(
    page_title="Multimodal LLM Code Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Multimodal AI Code Generator")
st.markdown("Generate code from natural language prompts and uploaded files!")
st.markdown("---")

# Initialize AI components
st.sidebar.header("⚙ Settings")
chat_model = st.sidebar.selectbox("Chat / Reasoning model", CHAT_MODELS, index=0)
code_model = st.sidebar.selectbox("Code‑generation model", CODE_MODELS, index=0)
agent, output_pipeline = initialize_ai_components(chat_model, code_model)

# Initialize session state for file tracking
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Add file uploader section
st.sidebar.markdown("---")
st.sidebar.header("📁 Reference Files")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF/code file(s) for the AI to reference",
    type=["pdf", "py", "js", "html", "css", "java", "cpp", "txt"],
)

# Handle file upload
if uploaded_file:
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save the file
    file_path = os.path.join(data_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Add to session state if not already there
    if uploaded_file.name not in st.session_state.uploaded_files:
        st.session_state.uploaded_files.append(uploaded_file.name)
        st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully!")

# Initialize session state for storing history
if 'history' not in st.session_state:
    st.session_state.history = []

with st.form("prompt_form"):
    prompt = st.text_area("Enter your prompt:",
                          height=100,
                          placeholder="Example: Read the contents of test.py and write a python script that calls the "
                                      "post endpoint to make a new item")
    submitted = st.form_submit_button("Generate Response")

# Process when form is submitted
if submitted and prompt:
    # Create a placeholder for displaying progress
    retry_placeholder = st.empty()
    progress_placeholder = st.empty()
    status_container = st.container()

    with st.spinner("Generating code..."):
        max_retries = 3
        retries = 0
        success = False
        error_context = ""
        retry_prompt = prompt
        while retries < max_retries and not success:
            try:
                if retries > 0:
                    retry_placeholder.warning(f"Retry attempt {retries}/{max_retries}...")
                    retry_prompt = (f"Original request: {prompt}  \nPrevious attempt failed with the following error:  \n{error_context[:400]}...  \nPlease generate a correct solution that avoids this error.")
                    with status_container:
                        st.info(f"------------- Previous attempt failed with error:  \n{error_context[:400]}...  \n------------- Retrying with this new knowledge.")
                else:
                    progress_placeholder.info("Querying AI agent...")

                # Get result from agent
                result = agent.query(retry_prompt)
                progress_placeholder.info("Processing response...")

                # Get formatted result and parse JSON
                next_result = output_pipeline.run(response=result)
                cleaned_json = str(next_result).replace("assistant:", "").strip()

                # Check if result was in JSON format
                try:
                    cleaned_json = json.loads(cleaned_json)
                    is_json = True
                except Exception:
                    is_json = False

                progress_placeholder.info("Displaying results...")

                # Display the results if output was in desired JSON format or (maybe) retry again
                if is_json:
                    st.subheader("Response")
                    st.markdown(f"**Description:** {cleaned_json['description']}")
                    st.code(cleaned_json['code'], language="python")
                    st.markdown(f"**Filename:** {cleaned_json['filename']}")
                    # Add a download button
                    st.download_button(
                        label=f"Download {cleaned_json['filename']}",
                        data=f"'''{cleaned_json['description']}'''\n{cleaned_json['code']}",
                        file_name=cleaned_json['filename'],
                        mime="text/plain"
                    )
                elif retries < 2:
                    raise ValueError("Response not in desired format.")
                else:
                    st.subheader("Response")
                    st.warning("Unable to generate a structured response. Loading raw response.")
                    st.write(cleaned_json)

                status_container.empty()
                success = True
                progress_placeholder.success("Code Generated Successfully!")

                try:
                    if not os.path.exists('output'):
                        os.makedirs('output')

                    with open(os.path.join("output", cleaned_json['filename']), "w") as file:
                        file.write(f"'''\n{cleaned_json['description']}\n'''\n")
                        file.write(cleaned_json['code'])
                    st.success(f"Code saved to 'output/{cleaned_json['filename']}'")
                except Exception as e:
                    st.error(f"There was an error generating the file: {e[:200]}...")

                # Add to history
                st.session_state.history.append(cleaned_json)

            except Exception as e:
                try:
                    with st.expander(f"Response from attempt {retries + 1}"):
                        st.code(cleaned_json)
                except:
                    try:
                        with st.expander(f"Response from attempt {retries + 1}"):
                            st.code(next_result)
                    except:
                        try:
                            with st.expander(f"Response from attempt {retries + 1}"):
                                st.code(result)
                        except:
                            pass
                retries += 1
                error_msg = str(e)
                error_traceback = traceback.format_exc()
                error_context = f"{error_msg}\n\nDetails: {error_traceback[-200:]}..."

                if retries >= max_retries:
                    progress_placeholder.error(f"Failed after {max_retries} attempts")
                    with status_container:
                        st.error(f"An error occurred: {error_msg} \nPlease try again with a different prompt.")
                        with st.expander("See detailed error"):
                            st.code(error_traceback)

# Display history if it isn't empty
if st.session_state.history:
    st.markdown("---")
    st.subheader("Model History")
    for i, entry in enumerate(st.session_state.history):
        try:
            with st.expander(f"#{i + 1}: {entry['filename']} - {entry['description']}"):
                st.code(entry['code'], language="python")
                if st.button(f"Download {entry['filename']}", key=f"download_{i}"):
                    st.download_button(
                        label=f"Download {entry['filename']}",
                        data=f"'''{entry['description']}'''\n{entry['code']}",
                        file_name=entry['filename'],
                        mime="text/plain",
                        key=f"download_button_{i}"
                    )
        except Exception as e:
            with st.expander(f"#{i + 1}: {entry[:200]}"):
                st.write(entry)

st.markdown("---")
st.markdown("Built with Ollama, LlamaIndex, LlamaCloud and Streamlit")