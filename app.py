"""
Cyber Bullying Detection Web Application
AUTHOR: S.Hemanth Chandra
"""

import streamlit as st
import torch
from transformers import pipeline
import plotly.graph_objects as go
from datetime import datetime

torch.set_num_threads(1)

st.set_page_config(
    page_title="Cyber Bullying Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stAlert { padding: 1rem; border-radius: 0.5rem; }
    h1 { color: #1f77b4; padding-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with st.spinner("🔄 Loading AI model... This may take a moment..."):
        classifier = pipeline(
            "text-classification",
            model="s-nlp/roberta_toxicity_classifier",
            top_k=None
        )
    return classifier

def predict_bullying(text, classifier, threshold=0.5):
    results = classifier(text)[0]
    scores = {r['label'].lower(): r['score'] for r in results}

    # s-nlp/roberta_toxicity_classifier outputs: neutral / toxic
    toxic_score = scores.get('toxic', 0.0)
    neutral_score = scores.get('neutral', 0.0)

    if toxic_score > threshold:
        is_bullying = True
        final_label = "BULLYING DETECTED"
    elif toxic_score > 0.35:
        is_bullying = True
        final_label = "POSSIBLY BULLYING"
    else:
        is_bullying = False
        final_label = "SAFE CONTENT"

    return {
        'is_bullying': is_bullying,
        'final_label': final_label,
        'max_toxic_score': toxic_score,
        'neutral_score': neutral_score,
        'all_scores': scores
    }

def create_gauge_chart(score, is_bullying):
    color = "#e74c3c" if is_bullying else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Toxicity Level", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 48}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 35], 'color': '#d5f4e6'},
                {'range': [35, 50], 'color': '#ffeaa7'},
                {'range': [50, 100], 'color': '#fab1a0'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
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
    st.title("🛡️ Cyber Bullying Detection System")
    st.markdown("### Real-time AI-powered content moderation")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.3,
            max_value=0.95,
            value=0.5,
            step=0.05,
            help="Higher threshold = fewer false positives"
        )
        st.markdown("---")
        st.header("ℹ️ About")
        st.info("""
        **Project:** Cyber Bullying Detection

        **AUTHOR:** S.Hemanth Chandra
        

        **Model:** s-nlp/roberta_toxicity_classifier

        Highly sensitive to mild insults, hate
        speech, and toxic language.
        """)
        st.markdown("---")
        st.header("🤖 Model Info")
        device_info = "GPU ✅" if torch.cuda.is_available() else "CPU ⚠️"
        st.success(f"**Device:** {device_info}")
        st.info(f"**Threshold:** {threshold:.2f}")

    try:
        classifier = load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📝 Enter Text to Analyze")
        user_input = st.text_area(
            "Type or paste your text here:",
            height=200,
            placeholder="Example: Hey, great job on your presentation today!",
        )
        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

    with col2:
        st.header("📊 Quick Examples")
        if st.button("✅ Safe Example", use_container_width=True):
            user_input = "Thanks for helping me with my homework. You're a great friend!"
            analyze_button = True
        if st.button("⚠️ Bullying Example", use_container_width=True):
            user_input = "I hate you, you are such a pig and nobody likes you."
            analyze_button = True
        if st.button("🔄 Neutral Example", use_container_width=True):
            user_input = "Let's meet at the library at 3 PM to study."
            analyze_button = True

    if analyze_button and user_input.strip():
        st.markdown("---")
        st.header("📈 Analysis Results")

        with st.spinner("🔄 Analyzing text..."):
            result = predict_bullying(user_input, classifier, threshold)

        res_col1, res_col2, res_col3 = st.columns([1, 1, 1])

        with res_col1:
            if result['final_label'] == "BULLYING DETECTED":
                st.error(f"### ⚠️ {result['final_label']}")
                st.markdown("""
                <div style='background-color:#ffe5e5;padding:1rem;border-radius:0.5rem;border-left:4px solid #e74c3c;'>
                    <strong>⚠️ Warning:</strong> This content contains cyber bullying or toxic language.
                </div>""", unsafe_allow_html=True)
            elif result['final_label'] == "POSSIBLY BULLYING":
                st.warning(f"### 🔶 {result['final_label']}")
                st.markdown("""
                <div style='background-color:#fff3cd;padding:1rem;border-radius:0.5rem;border-left:4px solid #f39c12;'>
                    <strong>🔶 Caution:</strong> This content may contain mildly toxic language.
                </div>""", unsafe_allow_html=True)
            else:
                st.success(f"### ✅ {result['final_label']}")
                st.markdown("""
                <div style='background-color:#e5ffe5;padding:1rem;border-radius:0.5rem;border-left:4px solid #2ecc71;'>
                    <strong>✅ Safe:</strong> This content appears appropriate and respectful.
                </div>""", unsafe_allow_html=True)

        with res_col2:
            st.metric(
                label="Toxicity Score",
                value=f"{result['max_toxic_score']*100:.1f}%",
                delta=f"{result['max_toxic_score']*100 - threshold*100:.1f}% vs threshold"
            )

        with res_col3:
            st.metric(
                label="Neutral Score",
                value=f"{result['neutral_score']*100:.1f}%"
            )

        st.plotly_chart(
            create_gauge_chart(result['max_toxic_score'], result['is_bullying']),
            use_container_width=True
        )

        st.subheader("📊 Score Breakdown")
        score_col1, score_col2 = st.columns(2)

        with score_col1:
            st.markdown(f"""
            <div style='background-color:#ffe5e5;padding:1rem;border-radius:0.5rem;'>
                <strong>🔴 Toxic Score</strong><br>
                <span style='font-size:2rem;font-weight:bold;'>{result['max_toxic_score']*100:.2f}%</span>
            </div>""", unsafe_allow_html=True)

        with score_col2:
            st.markdown(f"""
            <div style='background-color:#e5ffe5;padding:1rem;border-radius:0.5rem;'>
                <strong>🟢 Neutral Score</strong><br>
                <span style='font-size:2rem;font-weight:bold;'>{result['neutral_score']*100:.2f}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("### Toxicity Confidence Bar")
        st.progress(result['max_toxic_score'])

        st.markdown("---")
        st.subheader("📄 Analyzed Text")
        st.text_area("", value=user_input, height=100, disabled=True)
        st.caption(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    elif analyze_button and not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;padding:2rem;color:#666;'>
        <p><strong>Cyber Bullying Detection System</strong> | Powered by Toxicity Classifier</p>
        <p>Created by S.Hemanth Chandra </p>
        <p style='font-size:0.8rem;'>Always use human judgment for final moderation decisions.</p>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
