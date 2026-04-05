#  Cyber Bullying Detection System

**author:** S. Hemanth Chandra

---

##  Project Overview

The **Cyber Bullying Detection System** is a real-time AI-powered web application designed to detect toxic, harmful, and bullying content in text. Built using a fine-tuned **RoBERTa** transformer model and deployed via **Streamlit**, the system provides instant feedback on whether a given piece of text contains cyberbullying or toxic language.

This project addresses the growing problem of online harassment and cyberbullying by leveraging state-of-the-art Natural Language Processing (NLP) to automatically flag harmful content — helping platforms, educators, and individuals moderate digital communication more effectively.

---
##  Live Demo
 https://cyber-bullying-detection.streamlit.app/
##  Objectives

- Detect cyberbullying and toxic language in real-time from user-entered text
- Provide confidence scores and visual feedback for predictions
- Deploy an interactive, user-friendly web interface accessible without technical knowledge
- Use a pre-trained transformer model to achieve high accuracy without training from scratch

---

##  Model Details

| Property | Details |
|---|---|
| **Model** | `s-nlp/roberta_toxicity_classifier` |
| **Base Architecture** | RoBERTa (Robustly Optimized BERT Pretraining Approach) |
| **Task** | Binary Text Classification (Toxic / Neutral) |
| **Source** | HuggingFace Model Hub |
| **Why RoBERTa?** | Outperforms BERT on most NLP benchmarks; better handling of short, informal text like social media messages |

### Why this model over alternatives?

| Model | "i hate you" | "you are a pig" | "you are stupid" |
|---|---|---|---|
| `unitary/toxic-bert` | ~20% ❌ | ~30% ❌ | ~15% ❌ |
| `martin-ha/toxic-comment-model` | ~75% ⚠️ | ~78% ✅ | ~80% ✅ |
| `s-nlp/roberta_toxicity_classifier` | ~95% ✅ | ~92% ✅ | ~90% ✅ |

---

##  Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | Streamlit |
| **ML Framework** | HuggingFace Transformers, PyTorch |
| **Visualization** | Plotly |
| **Language** | Python 3.8+ |
| **Model Hosting** | HuggingFace Hub (auto-downloaded) |

---

##  Project Structure
cyber-bullying-detection/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
---

## 🔍 Classification Logic

| Toxic Score | Label |
|---|---|
| > 0.50 (threshold) | ⚠️ BULLYING DETECTED |
| 0.35 – 0.50 | 🔶 POSSIBLY BULLYING |
| < 0.35 | ✅ SAFE CONTENT |

---

## 📸 Sample Output

| Input Text | Result |
|---|---|
| "You're so stupid, nobody likes you" | ⚠️ BULLYING DETECTED (91%) |
| "I hate you, you are a pig" | ⚠️ BULLYING DETECTED (94%) |
| "Let's study together at the library" | ✅ SAFE CONTENT (4%) |
| "That's a bit rude honestly" | 🔶 POSSIBLY BULLYING (38%) |

---

## ⚠️ Limitations

- The model works best with **English text** — accuracy may drop for other languages or transliterated text (e.g., Telugu/Hindi written in English)
- Very **short or sarcastic** text may occasionally be misclassified
- This is an AI tool — **human moderation** should always be the final decision maker
- First run requires internet to **download the model** from HuggingFace

---

## 🔮 Future Improvements

- Add support for **multilingual** text (Hindi, Telugu, etc.)
- Batch analysis — upload a **CSV file** of comments for bulk detection
- Add **category-level** detection (threat, insult, hate speech, obscenity)
- Export analysis **reports as PDF**
- Deploy on **Streamlit Cloud / Hugging Face Spaces** for public access

---

## 📚 References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [s-nlp/roberta_toxicity_classifier](https://huggingface.co/s-nlp/roberta_toxicity_classifier)
- [Streamlit Documentation](https://docs.streamlit.io)
- [RoBERTa Paper — Liu et al., 2019](https://arxiv.org/abs/1907.11692)

---

## 👨‍💻 Author

**S. Hemanth Chandra**


---

*This project was developed for academic purposes as part of a Machine Learning / NLP course.*
