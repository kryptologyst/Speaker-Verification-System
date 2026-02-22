"""Streamlit demo for speaker verification."""

import streamlit as st
import torch
import numpy as np
import librosa
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import tempfile
import os

from src.speaker_verification.models import SpeakerVerifier
from src.speaker_verification.utils import get_device


# Page configuration
st.set_page_config(
    page_title="Speaker Verification Demo",
    page_icon="🎤",
    layout="wide"
)

# Privacy disclaimer
st.warning("""
**PRIVACY AND ETHICS DISCLAIMER**

This is a RESEARCH and EDUCATIONAL demonstration only. This system is NOT intended for production biometric identification.

- No raw personally identifiable information (PII) is logged or stored
- Voice cloning, impersonation, or biometric misuse is strictly prohibited
- Users must comply with applicable laws and ethical guidelines
- This software is provided for academic research and educational purposes only
""")

# Title
st.title("🎤 Speaker Verification System")
st.markdown("A modern speaker verification system implementing MFCC, x-vector, and ECAPA-TDNN models.")

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["mfcc", "xvector", "ecapa_tdnn"],
    help="Choose the speaker verification model to use"
)

# Threshold
threshold = st.sidebar.slider(
    "Verification Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Similarity threshold for verification"
)

# Initialize verifier
@st.cache_resource
def load_verifier(model_type: str):
    """Load speaker verifier."""
    config = {
        "sample_rate": 16000,
        "threshold": threshold
    }
    
    if model_type == "mfcc":
        config.update({
            "n_mfcc": 13,
            "n_fft": 2048,
            "hop_length": 512,
            "n_mels": 80
        })
    elif model_type == "xvector":
        config.update({
            "input_dim": 40,
            "hidden_dim": 512,
            "embedding_dim": 512,
            "dropout": 0.5
        })
    elif model_type == "ecapa_tdnn":
        config.update({
            "input_dim": 80,
            "channels": 512,
            "embedding_dim": 192,
            "dropout": 0.5
        })
    
    return SpeakerVerifier(model_type=model_type, config=config)

verifier = load_verifier(model_type)

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Verification", "👥 Speaker Database", "📊 Analysis"])

with tab1:
    st.header("Speaker Verification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Enroll Speaker")
        
        # Speaker ID input
        speaker_id = st.text_input("Speaker ID", placeholder="Enter unique speaker identifier")
        
        # Audio upload for enrollment
        enrollment_audio = st.file_uploader(
            "Upload Enrollment Audio",
            type=['wav', 'mp3', 'flac'],
            help="Upload audio file for speaker enrollment"
        )
        
        if st.button("Enroll Speaker") and speaker_id and enrollment_audio:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(enrollment_audio.read())
                    tmp_path = tmp_file.name
                
                # Enroll speaker
                verifier.enroll_speaker(speaker_id, tmp_path)
                
                # Clean up
                os.unlink(tmp_path)
                
                st.success(f"Speaker '{speaker_id}' enrolled successfully!")
                
            except Exception as e:
                st.error(f"Error enrolling speaker: {str(e)}")
    
    with col2:
        st.subheader("Verify Speaker")
        
        # Test audio upload
        test_audio = st.file_uploader(
            "Upload Test Audio",
            type=['wav', 'mp3', 'flac'],
            help="Upload audio file for verification"
        )
        
        # Speaker selection
        enrolled_speakers = list(verifier.speaker_database.keys())
        if enrolled_speakers:
            selected_speaker = st.selectbox("Select Speaker to Verify", enrolled_speakers)
        else:
            selected_speaker = None
            st.info("No speakers enrolled yet. Please enroll a speaker first.")
        
        if st.button("Verify Speaker") and test_audio and selected_speaker:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(test_audio.read())
                    tmp_path = tmp_file.name
                
                # Verify speaker
                result = verifier.verify_speaker(selected_speaker, tmp_path, threshold=threshold)
                
                # Clean up
                os.unlink(tmp_path)
                
                # Display results
                st.subheader("Verification Results")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Similarity", f"{result['similarity']:.4f}")
                with col_b:
                    st.metric("Threshold", f"{result['threshold']:.4f}")
                with col_c:
                    st.metric("Verified", "✅ Yes" if result['verified'] else "❌ No")
                
                # Confidence
                st.metric("Confidence", f"{result['confidence']:.4f}")
                
                # Visual indicator
                if result['verified']:
                    st.success("🎉 Speaker verification successful!")
                else:
                    st.error("🚫 Speaker verification failed!")
                
            except Exception as e:
                st.error(f"Error verifying speaker: {str(e)}")

with tab2:
    st.header("Speaker Database")
    
    # Database info
    db_info = verifier.get_speaker_database()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Speakers", db_info['num_speakers'])
    with col2:
        st.metric("Model Type", db_info['model_type'].upper())
    with col3:
        st.metric("Current Threshold", f"{db_info['threshold']:.3f}")
    
    # Speaker list
    if db_info['speaker_ids']:
        st.subheader("Enrolled Speakers")
        for speaker_id in db_info['speaker_ids']:
            st.write(f"• {speaker_id}")
    else:
        st.info("No speakers enrolled yet.")
    
    # Database management
    st.subheader("Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Database"):
            try:
                verifier.save_database("speaker_database.json")
                st.success("Database saved successfully!")
            except Exception as e:
                st.error(f"Error saving database: {str(e)}")
    
    with col2:
        if st.button("Load Database"):
            try:
                verifier.load_database("speaker_database.json")
                st.success("Database loaded successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error loading database: {str(e)}")

with tab3:
    st.header("Analysis")
    
    # Model information
    st.subheader("Model Information")
    model_info = verifier.model.get_model_info()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Model Type:** {model_info['model_type']}")
        st.write(f"**Device:** {model_info['device']}")
    with col2:
        st.write(f"**Total Parameters:** {model_info['total_parameters']:,}")
        st.write(f"**Trainable Parameters:** {model_info['trainable_parameters']:,}")
    
    # Performance metrics (if available)
    st.subheader("Performance Metrics")
    
    # Placeholder for metrics
    if db_info['num_speakers'] > 0:
        st.info("Performance metrics will be available after evaluation on test data.")
        
        # Example metrics (would be computed from actual evaluation)
        metrics_data = {
            "Metric": ["EER", "AUC", "Accuracy", "Min DCF"],
            "Value": [0.123, 0.987, 0.945, 0.089],
            "Description": [
                "Equal Error Rate (lower is better)",
                "Area Under ROC Curve (higher is better)",
                "Verification Accuracy (higher is better)",
                "Minimum Detection Cost Function (lower is better)"
            ]
        }
        
        st.dataframe(metrics_data, use_container_width=True)
    else:
        st.info("No speakers enrolled yet. Please enroll speakers to see performance metrics.")
    
    # Visualization
    st.subheader("Visualization")
    
    # Example similarity distribution
    if db_info['num_speakers'] > 0:
        # Generate example data
        np.random.seed(42)
        same_speaker_scores = np.random.normal(0.8, 0.1, 100)
        different_speaker_scores = np.random.normal(0.3, 0.1, 100)
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=same_speaker_scores,
            name="Same Speaker",
            opacity=0.7,
            nbinsx=20
        ))
        
        fig.add_trace(go.Histogram(
            x=different_speaker_scores,
            name="Different Speaker",
            opacity=0.7,
            nbinsx=20
        ))
        
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"Threshold: {threshold}")
        
        fig.update_layout(
            title="Similarity Score Distribution",
            xaxis_title="Similarity Score",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Visualization will be available after enrolling speakers.")

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer:** This is a research demonstration. Not for production use. 
Voice cloning and biometric misuse are prohibited. Use responsibly and ethically.
""")
