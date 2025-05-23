"""
Manages user feedback collection and visualization for generated code.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from feedback_analyzer import FeedbackAnalyzer

class FeedbackManager:
    """
    A class to collect, store, and analyze user feedback on generated code.
    """

    def __init__(self, log_path: str = "logs/user_feedback.json"):
        """
        Initialize the FeedbackManager.

        Args:
            log_path: Path to store feedback logs
        """
        self.log_path = log_path
        self._ensure_log_file_exists()

    def _ensure_log_file_exists(self):
        """Ensure the log file and directory exist."""
        log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                json.dump([], f)

    def is_feedback_recorded(self, code_id: str) -> bool:
        """
        Check if feedback has already been recorded for a specific code ID.

        Args:
            code_id: Identifier for the code to check

        Returns:
            bool: True if feedback has already been recorded for this code ID
        """
        try:
            with open(self.log_path, 'r') as f:
                feedbacks = json.load(f)

            # Check if any feedback entry has this code_id
            for entry in feedbacks:
                if entry.get("code_id") == code_id:
                    return True
            return False
        except Exception as e:
            return False

    def record_feedback(self,
                        feedback_rating: int,
                        code_id: str,
                        feedback_comment: Optional[str] = None,
                        chat_model: Optional[str] = None,
                        code_model: Optional[str] = None,
                        code: Optional[str] = None,
                        prompt: Optional[str] = None,
                        code_description: Optional[str] = None) -> bool:
        """
        Record user feedback for a generated code.

        Args:
            feedback_rating: User rating (1-5)
            code_id: Identifier for the code being rated
            feedback_comment: Optional user comment
            chat_model: Chat model used
            code_model: Code model used
            code: The generated code being rated
            prompt: The prompt that generated the code
            code_description: Description of what the code does

        Returns:
            bool: True if feedback was successfully recorded
        """
        # Check if feedback already exists for this code_id
        if self.is_feedback_recorded(code_id):
            return True

        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "code_id": code_id,
            "rating": feedback_rating,
            "comment": feedback_comment if feedback_comment else "",
            "chat_model": chat_model,
            "code_model": code_model,
            "code": code,
            "prompt": prompt,
            "code_description": code_description
        }

        try:
            with open(self.log_path, 'r') as f:
                feedbacks = json.load(f)
            feedbacks.append(feedback_entry)
            with open(self.log_path, 'w') as f:
                json.dump(feedbacks, f, indent=2)
            return True
        except Exception as e:
            return False

    @classmethod
    def load_feedback(cls) -> List[Dict[str, Any]]:
        """Load all saved feedback."""
        log_path = "logs/user_feedback.json"
        if not os.path.exists(log_path):
            return []

        try:
            with open(log_path, 'r') as f:
                return json.load(f)
        except Exception:
            return []


def render_feedback_dashboard(model: str = "mistral"):
    """Render the feedback analysis dashboard in Streamlit."""
    st.header("User Feedback Dashboard")
    feedbacks = FeedbackManager.load_feedback()

    if not feedbacks:
        st.info("No user feedback has been collected yet.")
        return

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(feedbacks)

    # Overall metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_rating = df["rating"].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}/5")
    with col2:
        total_feedback = len(df)
        st.metric("Total Feedback Collected", total_feedback)
    with col3:
        positive_feedback = len(df[df["rating"] >= 4])
        positive_percent = (positive_feedback / total_feedback) * 100 if total_feedback > 0 else 0
        st.metric("Positive Feedback", f"{positive_percent:.1f}%")

    # Add LLM Analysis Section
    st.subheader("AI-Powered Feedback Analysis")    
    analyzer = FeedbackAnalyzer(model)
    
    analysis_tab, categories_tab, suggestions_tab = st.tabs([
        "Feedback Insights", 
        "Feedback Categories",
        "Improvement Suggestions"
    ])
    
    with analysis_tab:
        if st.button("Generate insights about the feedbacks given by users"):
            with st.spinner("Analyzing feedback..."):
                analysis_results = analyzer.analyze_feedback(feedbacks)
                try:
                    if "error" in analysis_results:
                        st.error(f"Analysis failed: {analysis_results['error']}")
                    else:
                        # Display common themes
                        if analysis_results.get("common_themes", []):   
                            st.write("### Common Themes")
                            for theme in analysis_results.get("common_themes", []):
                                st.write(f"- {theme}")

                        # Display areas for improvement
                        if analysis_results.get("areas_for_improvement", []):
                            st.write("### Areas for Improvement")
                            for area in analysis_results.get("areas_for_improvement", []):
                                st.write(f"- {area}")
                            
                        # Display what users like
                        if analysis_results.get("what_users_like", []):
                            st.write("### What Users Like")
                            for like in analysis_results.get("what_users_like", []):
                                st.write(f"- {like}")
                            
                        # Display suggestions
                        if analysis_results.get("suggestions", []):
                            st.write("### Suggestions for Improvement")
                            for suggestion in analysis_results.get("suggestions", []):
                                st.write(f"- {suggestion}")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    
    with categories_tab:
        if st.button("Categorize the feedbacks given by users"):
            with st.spinner("Categorizing feedback..."):
                categories = analyzer.categorize_feedback(feedbacks)
                for category, comments in categories.items():
                    try:    
                        with st.expander(f"{category} ({len(comments)})"):
                            for comment in comments:
                                st.write(f"- Code ID: {comment['code_id']}, Rating: {comment['rating']}, Comment: {comment['comment'] if 'comment' in comment and comment['comment'] else 'No comment'}")
                    except Exception as e:
                        st.error(f"Categorization failed: {e}")
    
    with suggestions_tab:
        if feedbacks:
            selected_feedback = st.selectbox("**Select feedback to analyze**",
                                             options=range(len(feedbacks)),
                                             format_func=lambda x: f"Rating: {feedbacks[x]['rating']} - Comment: {feedbacks[x]['comment'][:50] + '...' if feedbacks[x]['comment'] else 'No comment'}")
            
            if st.button("Generate suggestions for improvement of the selected feedback"):
                with st.spinner("Generating suggestions..."):
                    feedback_entry = feedbacks[selected_feedback]
                    suggestions = analyzer.generate_improvement_suggestions(feedback_entry.get("code", ""), feedback_entry.get("code_description", ""), feedback_entry.get("comment", ""), feedback_entry.get("prompt", ""))
                    try:
                        if "error" in suggestions:
                            st.error(f"An error occurred while generating suggestions. Please try again.  \nError: {suggestions['error']}")
                        else:
                            st.write("### Suggested Improvements")
                            for suggestion in suggestions['suggestions']:
                                st.write(f"- {suggestion}")
                    except Exception as e:
                        st.error(f"An error occurred while generating suggestions. Please try again.  \nError Summary: {e}")

    st.markdown("---")
    st.subheader("Rating Distribution")
    rating_counts = df["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(rating_counts.index, rating_counts.values, color='skyblue')
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ratings")
    ax.set_xticks(range(1, 6))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    st.pyplot(fig)

    # Model comparison
    if "chat_model" in df.columns and "code_model" in df.columns:
        st.subheader("Model Performance by Rating")
        tab1, tab2 = st.tabs(["Chat Models", "Code Models"])

        with tab1:
            _plot_model_ratings(df, "chat_model")

        with tab2:
            _plot_model_ratings(df, "code_model")

    # Recent feedback
    st.subheader("Recent Feedback")
    recent_df = df.sort_values("timestamp", ascending=False).head(10)

    for i, (_, row) in enumerate(recent_df.iterrows()):
        with st.expander(f"{row['timestamp']} - Rating: {'⭐' * int(row['rating'])}"):
            st.write(f"**Code ID:** {row['code_id']}")
            if "chat_model" in row and "code_model" in row:
                st.write(f"**Models:** Chat: {row['chat_model']} / Code: {row['code_model']}")
            if row['comment']:
                st.write(f"**Comment:** {row['comment']}")


def _plot_model_ratings(df: pd.DataFrame, model_col: str):
    """Plot average ratings for different models."""
    if model_col not in df.columns:
        st.info(f"No {model_col} data available.")
        return

    # Filter out rows where model info is missing
    df_clean = df[df[model_col].notna()]

    if df_clean.empty:
        st.info(f"No {model_col} data available.")
        return

    # Average rating by model
    ratings_by_model = df_clean.groupby(model_col)["rating"].mean()
    ratings_count = df_clean.groupby(model_col)["rating"].count()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ratings_by_model.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 5)  # Set y-axis to range from 0 to 5
    ax.set_xlabel(f"{model_col.replace('_', ' ').title()}(s)")
    ax.set_ylabel("Average Rating")
    ax.set_title(f"Average Rating by {model_col.replace('_', ' ').title()}")

    # Add count labels
    for i, v in enumerate(ratings_by_model):
        ax.text(i, v + 0.1, f"n={ratings_count[ratings_by_model.index[i]]}",
                ha='center', va='bottom', fontsize=9)
    st.pyplot(fig)