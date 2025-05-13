import json
import os
import traceback
import uuid

import streamlit as st

from main import initialize_ai_components
from model_registry import CHAT_MODELS, CODE_MODELS
from model_evaluator import render_evaluation_dashboard
from feedback_manager import FeedbackManager, render_feedback_dashboard

# Set page configuration
st.set_page_config(
    page_title="Multimodal LLM Code Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Multimodal AI Code Generator")
st.markdown("Generate code from natural language prompts and uploaded files!")

# Initialize AI components
st.sidebar.header("⚙ Settings")
chat_model = st.sidebar.selectbox("Chat / Reasoning model", CHAT_MODELS, index=0)
code_model = st.sidebar.selectbox("Code‑generation model", CODE_MODELS, index=0)
agent, output_pipeline, model_evaluator = initialize_ai_components(chat_model, code_model)

# Initialize feedback manager
feedback_manager = FeedbackManager()

# Initialize session state for file tracking
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Initialize session state for IDs
if 'code_ids' not in st.session_state:
    st.session_state.code_ids = {}

# Initialize session state for storing history
if 'history' not in st.session_state:
    st.session_state.history = []

# Initialize session state for tracking current feedback
if 'current_code_id' not in st.session_state:
    st.session_state.current_code_id = None

# Initialize session state for tracking if feedback was submitted
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False


# Function to handle feedback submission
def submit_feedback(rating, code_id, comment, chat_model, code_model, code=None, prompt=None, description=None):
    feedback_success = feedback_manager.record_feedback(
        rating, code_id, comment, chat_model, code_model, code=code, prompt=prompt, code_description=description
    )
    if feedback_success:
        st.session_state.feedback_submitted = True
    return feedback_success


# Add file uploader section
st.sidebar.markdown("---")
st.sidebar.header("Reference Files")
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

# Create tabs for generation, evaluation, and feedback
tab1, tab2, tab3 = st.tabs(["Code Generation", "Model Evaluation", "User Feedback"])

with tab1:
    with st.form("prompt_form"):
        prompt = st.text_area("Enter your prompt:",
                              height=100,
                              placeholder="Example: Read the contents of test.py and write a python script that calls the "
                                          "post endpoint to make a new item")
        submitted = st.form_submit_button("Generate Response")

    # Process when form is submitted
    if submitted and prompt:
        # Reset feedback state for new generation
        st.session_state.feedback_submitted = False

        # Create a placeholder for displaying progress
        retry_placeholder = st.empty()
        progress_placeholder = st.empty()
        status_container = st.container()

        # Start the evaluation
        model_evaluator.start_evaluation(chat_model, code_model, prompt)

        with st.spinner("Generating code..."):
            max_retries = 3
            retries = 0
            success = False
            error_context = ""
            retry_prompt = prompt
            while retries < max_retries and not success:
                try:
                    if retries > 0:
                        retry_prompt = f"Original request: {prompt}  \nPrevious attempt failed with the following error:  \n{error_context[:400]}...  \nPlease generate a correct solution that avoids this error to respond to original request."
                        with status_container:
                            st.info(f"**Previous attempt failed with error:**  \n{error_context[:400]}...  \n**Retrying with this new knowledge.**")

                        # Record retry in evaluation
                        model_evaluator.record_retry(error_context)

                    # Get result from agent
                    progress_placeholder.info("Querying AI agent...")
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
                        if cleaned_json['code']:
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

                        # Record partial success in evaluation
                        model_evaluator.record_success({"code": str(cleaned_json)})

                    # Record success in evaluation
                    completion_time = model_evaluator.record_success(cleaned_json)

                    # Generate a unique ID for this code generation
                    code_id = f"code_{uuid.uuid4().hex[:8]}"
                    st.session_state.code_ids[len(st.session_state.history)] = code_id
                    st.session_state.current_code_id = code_id  # Store current code ID

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

                    # Show completion time
                    st.info(f"Code generated in {completion_time:.2f} seconds")

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
                    retry_placeholder.warning(f"Retry attempt {retries}/{max_retries}...")

                    if retries >= max_retries:
                        progress_placeholder.error(f"Failed after {max_retries} attempts")
                        with status_container:
                            st.error(f"**An error occurred:** {error_msg[:300]}  \n**Please try again with a different prompt.**")
                            with st.expander("See detailed error"):
                                st.code(error_traceback)

                        # Record failure in evaluation
                        model_evaluator.record_failure(error_context)

    # Display feedback section for the current result
    if st.session_state.current_code_id and not st.session_state.feedback_submitted:
        if not feedback_manager.is_feedback_recorded(st.session_state.current_code_id):
            st.markdown("---")
            st.markdown("#### Was this response helpful?")
            feedback_comment = st.text_area("Additional comments (optional):", key="current_feedback_comment")

            col1, col2, col3, col4, col5 = st.columns(5)
            for i, col in enumerate([col1, col2, col3, col4, col5], 1):
                if col.button(f"{i * '⭐'}", key=f"current_rating_{i}"):
                    # Get the current code details from the last result
                    current_code = None
                    current_prompt = None
                    current_description = None
                    if st.session_state.history:
                        last_result = st.session_state.history[-1]
                        current_code = last_result.get('code')
                        current_description = last_result.get('description')
                        
                    feedback_success = submit_feedback(
                        i, st.session_state.current_code_id, feedback_comment,
                        chat_model, code_model,
                        code=current_code,
                        prompt=prompt,  # Using the prompt from the form
                        description=current_description
                    )
                    if feedback_success:
                        st.success("Thank you for your feedback!")
                    else:
                        st.error("Error recording feedback. Please try again.")
        else:
            st.markdown("---")
            st.success("Feedback already recorded for this response. Thank you!")

    # Display history if it isn't empty
    if st.session_state.history:
        st.markdown("---")
        st.subheader("Model History")
        for i, entry in enumerate(st.session_state.history):
            try:
                # Get code ID or create one if it doesn't exist
                if i not in st.session_state.code_ids:
                    st.session_state.code_ids[i] = f"history_{uuid.uuid4().hex[:8]}"
                code_id = st.session_state.code_ids[i]

                with st.expander(f"#{i + 1}: {entry['filename']} - {entry['description']}"):
                    st.code(entry['code'], language="python")
                    st.download_button(
                        label=f"Download {entry['filename']}",
                        data=f"'''{entry['description']}'''\n{entry['code']}",
                        file_name=entry['filename'],
                        mime="text/plain",
                        key=f"download_button_{i}"
                    )

                    # Add feedback for historical items
                    if feedback_manager.is_feedback_recorded(code_id):
                        st.success("Feedback already recorded for this response.")
                    else:
                        st.write("Rate this code:")
                        feedback_cols = st.columns(5)
                        feedback_comment = st.text_area("Additional comments (optional):", key=f"history_comment_{i}")
                        for j, fcol in enumerate(feedback_cols, 1):
                            if fcol.button(f"{j * '⭐'}", key=f"history_rating_{j}_{code_id}"):
                                feedback_success = submit_feedback(
                                    j, code_id, feedback_comment,
                                    chat_model, code_model,
                                    code=entry['code'],
                                    prompt='Unavailable',
                                    description=entry['description']
                                )
                                if feedback_success:
                                    st.success("Feedback recorded!")
                                else:
                                    st.error("Error recording feedback. Please try again.")
            except Exception as e:
                with st.expander(f"#{i + 1}: {entry[:200]}"):
                    st.write(entry)

with tab2:
    # Render the evaluation dashboard
    render_evaluation_dashboard()

with tab3:
    # Render the feedback dashboard
    render_feedback_dashboard()

st.markdown("---")
st.markdown("Built with Ollama, LlamaIndex, LlamaCloud and Streamlit")