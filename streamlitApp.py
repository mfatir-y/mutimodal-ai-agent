import json
import os
import streamlit as st

from main import initialize_ai_components

# Set page configuration
st.set_page_config(
    page_title="Multimodal LLM Code Generator",
    layout="wide"
)

st.title("Multimodal AI Code Generator")
st.markdown("Enter a prompt and let the AI generate code for you!")
st.markdown("---")

# Initialize session state for storing history
if 'history' not in st.session_state:
    st.session_state.history = []

# Initialize AI components
agent, output_pipeline = initialize_ai_components()

# Display history
if st.session_state.history:
    st.subheader("Model History")
    for i, entry in enumerate(st.session_state.history):
        with st.expander(f"#{i + 1}: {entry['filename']} - {entry['description'][:50]}..."):
            st.code(entry['code'], language="python")
            if st.button(f"Download {entry['filename']}", key=f"download_{i}"):
                st.download_button(
                    label=f"Download {entry['filename']}",
                    data=f"'''{entry['description']}'''\n{entry['code']}",
                    file_name=entry['filename'],
                    mime="text/plain",
                    key=f"download_button_{i}"
                )

# Input area
with st.form("prompt_form"):
    prompt = st.text_area("Enter your prompt:",
                          height=100,
                          placeholder="Example: read the contents of test.py and write a python script that calls the "
                                      "post endpoint to make a new item")
    submitted = st.form_submit_button("Generate Response")

# Process when form is submitted
if submitted and prompt:
    with st.spinner("Generating code..."):
        try:
            # Create a placeholder for displaying progress
            progress_placeholder = st.empty()
            progress_placeholder.info("Querying AI agent...")

            # Get result from agent
            result = agent.query(prompt)
            progress_placeholder.info("Processing response...")
            # Get formatted result and parse JSON
            next_result = output_pipeline.run(response=result)
            cleaned_json = str(next_result).replace("assistant:", "").strip()
            cleaned_json = json.loads(cleaned_json)

            # Success message
            progress_placeholder.success("Code generated successfully!")

            # Display the results
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

            try:
                with open(os.path.join("output", cleaned_json['filename']), "w") as file:
                    file.write(f"'''\n"
                               f"{cleaned_json['description']}\n"
                               f"'''\n")
                    file.write(cleaned_json['code'])
                st.success(f"Code saved to output/{cleaned_json['filename']}")
            except Exception as e:
                st.error(f"Error saving the file: {e}")

            # Add to history
            st.session_state.history.append(cleaned_json)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please try again with a different prompt.")

st.markdown("---")
st.markdown("Built with Ollama, LlamaIndex, LlamaCloud and Streamlit")