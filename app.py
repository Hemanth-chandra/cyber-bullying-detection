"""
Cyber Bullying Detection Web Application
Name: S.Hemanth Chandra
Reg No: 2025MS020

A Streamlit-based web application for real-time cyber bullying detection
using pre-trained BERT (unitary/toxic-bert) model.
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import torch
torch.set_num_threads(1)
# Page configuration
st.set_page_config(
    page_title="Cyber Bullying Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained BERT model and tokenizer."""
    model_name = "unitary/toxic-bert"
    
    with st.spinner("🔄 Loading AI model... This may take a moment..."):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
    
    return model, tokenizer, device

def predict_bullying(text, model, tokenizer, device, threshold=0.7):
    """
    Predict if the text contains cyber bullying.
    
    Args:
        text: Input text to analyze
        model: Pre-trained BERT model
        tokenizer: BERT tokenizer
        device: Computing device (CPU/GPU)
        threshold: Confidence threshold for classification
        
    Returns:
        Dictionary with prediction results
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Calculate probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    toxic_prob = probabilities[0][1].item()
    non_toxic_prob = probabilities[0][0].item()
    
    # Apply threshold
    is_bullying = toxic_prob >= threshold
    
    return {
        'is_bullying': is_bullying,
        'toxic_probability': toxic_prob,
        'non_toxic_probability': non_toxic_prob,
        'confidence': toxic_prob,
        'label': 'BULLYING DETECTED' if is_bullying else 'SAFE CONTENT'
    }

def create_gauge_chart(confidence, is_bullying):
    """Create a gauge chart showing confidence level."""
    color = "#e74c3c" if is_bullying else "#2ecc71"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Toxicity Level", 'font': {'size': 24}},
        number = {'suffix': "%", 'font': {'size': 48}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d5f4e6'},
                {'range': [30, 70], 'color': '#ffeaa7'},
                {'range': [70, 100], 'color': '#fab1a0'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def main():
    # Header
    st.title("🛡️ Cyber Bullying Detection System")
    st.markdown("### Real-time AI-powered content moderation using BERT")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Confidence threshold slider
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="Higher threshold = fewer false positives, but may miss some cases"
        )
        
        st.markdown("---")
        
        # About section
        st.header("ℹ️ About")
        st.info("""
        **Project:** Cyber Bullying Detection
        
        **Student:** S.Hemanth Chandra  
        **Reg No:** 2025MS020
        
        **Model:** Pre-trained BERT (unitary/toxic-bert)
        
        This system uses advanced AI to detect potentially harmful content in real-time.
        """)
        
        st.markdown("---")
        
        # Model info
        st.header("🤖 Model Info")
        device_info = "GPU ✅" if torch.cuda.is_available() else "CPU ⚠️"
        st.success(f"**Device:** {device_info}")
        st.info(f"**Threshold:** {threshold:.2f}")
    
    # Load model
    try:
        model, tokenizer, device = load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Text to Analyze")
        
        # Text input
        user_input = st.text_area(
            "Type or paste your text here:",
            height=200,
            placeholder="Example: Hey, great job on your presentation today!",
            help="Enter any text to check if it contains cyber bullying content"
        )
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    with col2:
        st.header("📊 Quick Examples")
        
        # Example buttons
        if st.button("✅ Safe Example", use_container_width=True):
            user_input = "Thanks for helping me with my homework. You're a great friend!"
            analyze_button = True
        
        if st.button("⚠️ Bullying Example", use_container_width=True):
            user_input = "You're so stupid, nobody likes you at all."
            analyze_button = True
        
        if st.button("🔄 Neutral Example", use_container_width=True):
            user_input = "Let's meet at the library at 3 PM to study."
            analyze_button = True
    
    # Analysis results
    if analyze_button and user_input.strip():
        st.markdown("---")
        st.header("📈 Analysis Results")
        
        with st.spinner("🔄 Analyzing text..."):
            # Get prediction
            result = predict_bullying(user_input, model, tokenizer, device, threshold)
        
        # Display results in columns
        res_col1, res_col2, res_col3 = st.columns([1, 1, 1])
        
        with res_col1:
            if result['is_bullying']:
                st.error(f"### ⚠️ {result['label']}")
                st.markdown("""
                <div style='background-color: #ffe5e5; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c;'>
                    <strong>⚠️ Warning:</strong> This content may contain cyber bullying or toxic language.
                    Recommended action: Review and moderate.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"### ✅ {result['label']}")
                st.markdown("""
                <div style='background-color: #e5ffe5; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2ecc71;'>
                    <strong>✅ Safe:</strong> This content appears to be appropriate and respectful.
                </div>
                """, unsafe_allow_html=True)
        
        with res_col2:
            st.metric(
                label="Toxicity Score",
                value=f"{result['toxic_probability']*100:.1f}%",
                delta=f"{result['toxic_probability']*100 - threshold*100:.1f}% vs threshold"
            )
        
        with res_col3:
            st.metric(
                label="Safety Score",
                value=f"{result['non_toxic_probability']*100:.1f}%",
                delta=None
            )
        
        # Gauge chart
        st.plotly_chart(
            create_gauge_chart(result['confidence'], result['is_bullying']),
            use_container_width=True
        )
        
        # Detailed probabilities
        st.subheader("📊 Detailed Probabilities")
        
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
                <h4 style='color: #2ecc71;'>Non-Toxic Probability</h4>
                <h2>{result['non_toxic_probability']*100:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with prob_col2:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
                <h4 style='color: #e74c3c;'>Toxic Probability</h4>
                <h2>{result['toxic_probability']*100:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bar
        st.markdown("### Confidence Visualization")
        st.progress(result['confidence'])
        
        # Analyzed text
        st.markdown("---")
        st.subheader("📄 Analyzed Text")
        st.text_area("", value=user_input, height=100, disabled=True)
        
        # Timestamp
        st.caption(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    elif analyze_button and not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #666;'>
        <p><strong>Cyber Bullying Detection System</strong> | Powered by Pre-trained BERT</p>
        <p>Created by S.Hemanth Chandra (2025MS020)</p>
        <p style='font-size: 0.8rem;'>This system uses AI to help identify potentially harmful content. 
        Always use human judgment for final moderation decisions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
