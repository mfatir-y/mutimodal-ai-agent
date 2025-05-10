import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path


class ModelEvaluator:
    """
    A class to evaluate and compare LLM model performance for code generation tasks.
    """

    def __init__(self, log_path: str = "logs/model_evaluations.json"):
        """
        Initialize the ModelEvaluator.

        Args:
            log_path: Path to store evaluation logs
        """
        self.log_path = log_path
        self._ensure_log_file_exists()
        self.current_evaluation = {
            "timestamp": None,
            "chat_model": None,
            "code_model": None,
            "prompt": None,
            "completion_time": None,
            "tokens_generated": None,
            "retry_count": 0,
            "success": False,
            "error": None,
            "code_metrics": {}
        }

    def _ensure_log_file_exists(self):
        """Ensure the log file and directory exist."""
        log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                json.dump([], f)

    def start_evaluation(self, chat_model: str, code_model: str, prompt: str):
        """
        Start tracking a new evaluation.

        Args:
            chat_model: Name of the chat/reasoning model
            code_model: Name of the code generation model
            prompt: User prompt for code generation
        """
        self.current_evaluation = {
            "timestamp": datetime.now().isoformat(),
            "chat_model": chat_model,
            "code_model": code_model,
            "prompt": prompt,
            "start_time": time.time(),
            "completion_time": None,
            "tokens_generated": None,
            "retry_count": 0,
            "success": False,
            "error": None,
            "code_metrics": {}
        }
        return self

    def record_retry(self, error: str):
        """Record a retry attempt and the associated error."""
        self.current_evaluation["retry_count"] += 1
        self.current_evaluation["error"] = error
        return self

    def record_success(self, code_output: Dict[str, Any]):
        """
        Record a successful code generation.
        Args: - code_output: The generated code output dictionary
        """
        self.current_evaluation["success"] = True
        self.current_evaluation["completion_time"] = time.time() - self.current_evaluation.pop("start_time")

        # Calculate approximate tokens generated (rough estimate)
        code_length = len(code_output.get("code", ""))
        self.current_evaluation["tokens_generated"] = code_length // 4  # Rough approximation

        # Calculate basic code metrics
        self.current_evaluation["code_metrics"] = self._calculate_code_metrics(code_output.get("code", ""))

        # Save the evaluation
        self._save_evaluation()

        return self.current_evaluation["completion_time"]

    def record_failure(self, error: str):
        """Record a final failure after all retries."""
        self.current_evaluation["success"] = False
        self.current_evaluation["completion_time"] = time.time() - self.current_evaluation.pop("start_time")
        self.current_evaluation["error"] = error

        # Save the evaluation
        self._save_evaluation()

        return self.current_evaluation["completion_time"]

    def _calculate_code_metrics(self, code: str) -> Dict[str, Any]:
        """
        Calculate basic metrics about the generated code.
        Args: - code: The generated code string
        Returns: Dictionary with code metrics
        """
        lines = code.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        metrics = {
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "character_count": len(code),
            "has_docstrings": '"""' in code or "'''" in code,
            "has_comments": any(line.strip().startswith("#") for line in lines),
            "avg_line_length": sum(len(line) for line in non_empty_lines) / max(len(non_empty_lines), 1)
        }

        return metrics

    def _save_evaluation(self):
        """Save the current evaluation to the log file."""
        try:
            with open(self.log_path, 'r') as f:
                evaluations = json.load(f)
            evaluations.append(self.current_evaluation)
            with open(self.log_path, 'w') as f:
                json.dump(evaluations, f, indent=2)
        except Exception as e:
            print(f"Error saving the evaluation: {e}")

    @classmethod
    def load_evaluations(cls) -> List[Dict[str, Any]]:
        """Load all saved evaluations."""
        log_path = "logs/model_evaluations.json"
        if not os.path.exists(log_path):
            return []

        try:
            with open(log_path, 'r') as f:
                return json.load(f)
        except Exception:
            return []


def render_evaluation_dashboard():
    """Render the model evaluation dashboard in Streamlit."""
    st.header("Model Evaluation Dashboard")
    evaluations = ModelEvaluator.load_evaluations()
    if not evaluations:
        st.info("No model evaluations have been recorded yet. Generate some code to see performance metrics!")
        return

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(evaluations)
    col1, col2 = st.columns(2)
    with col1:
        success_rate = df["success"].mean() * 100
        st.metric("Overall Success Rate", f"{success_rate:.1f}%")
    with col2:
        avg_completion_time = df["completion_time"].mean()
        st.metric("Average Completion Time", f"{avg_completion_time:.2f}s")

    # Model comparison
    st.subheader("Model Performance Comparison")
    # Get unique models
    chat_models = df["chat_model"].unique()
    code_models = df["code_model"].unique()

    # Create a tab for each type of model
    tab1, tab2 = st.tabs(["Chat Models", "Code Models"])
    with tab1:
        _plot_model_metrics(df, "chat_model")
    with tab2:
        _plot_model_metrics(df, "code_model")

    # Recent evaluations
    st.subheader("Recent Evaluations")
    recent_df = df.sort_values("timestamp", ascending=False).head(10)

    for _, row in recent_df.iterrows():
        with st.expander(f"{row['timestamp']} - {row['chat_model']}/{row['code_model']}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Prompt:** {row['prompt']}")
                st.write(f"**Success:** {'✅' if row['success'] else '❌'}")
                if not row['success']:
                    st.write(f"**Error:** {row['error']}")
                st.write(f"**Retries:** {row['retry_count']}")

            with col2:
                st.write(f"**Completion time:** {row['completion_time']:.2f}s")
                st.write(f"**Tokens (est.):** {row['tokens_generated']}")

                # Show code metrics if available
                if row['code_metrics']:
                    st.write("**Code Metrics:**")
                    for key, value in row['code_metrics'].items():
                        if isinstance(value, float):
                            st.write(f"- {key}: {value:.2f}")
                        else:
                            st.write(f"- {key}: {value}")


def _plot_model_metrics(df: pd.DataFrame, model_col: str):
    """Plot performance metrics for different models."""
    models = df[model_col].unique()

    # Success rate by model
    success_by_model = df.groupby(model_col)["success"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    success_by_model.plot(kind="bar", ax=ax)
    ax.set_xlabel(f"{model_col.replace('_', ' ').title()}(s)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(f"Success Rate by {model_col.replace('_', ' ').title()}")
    st.pyplot(fig)

    # Completion time by model
    time_by_model = df.groupby(model_col)["completion_time"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    time_by_model.plot(kind="bar", ax=ax)
    ax.set_xlabel(f"{model_col.replace('_', ' ').title()}(s)")
    ax.set_ylabel("Average Completion Time (s)")
    ax.set_title(f"Completion Time by {model_col.replace('_', ' ').title()}")
    st.pyplot(fig)