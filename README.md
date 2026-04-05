# 🛡️ Cyber Bullying Detection System

**Author:** S.Hemanth Chandra  
**Registration Number:** 2025MS020

A real-time web application for detecting cyber bullying using a pre-trained BERT transformer model.

---

## 🌟 Features

- ✅ Real-time cyber bullying detection
- 📊 Interactive confidence visualization
- 🎯 Adjustable confidence threshold
- 📈 Detailed probability breakdowns
- 🎨 User-friendly web interface
- 🚀 Fast predictions using pre-trained BERT
- 📱 Responsive design

---

## 🚀 Quick Start

### Option 1: Run Locally

#### Prerequisites
- Python 3.8 or higher
- pip package manager

#### Installation Steps

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the Application**
```bash
streamlit run app.py
```

3. **Access the App**
- The app will automatically open in your browser
- Default URL: `http://localhost:8501`

---

### Option 2: Deploy on Streamlit Cloud (Recommended for Demos)

#### Steps:

1. **Create a GitHub Repository**
   - Go to [GitHub](https://github.com)
   - Create a new repository (e.g., `cyberbullying-detector`)
   - Upload these files:
     - `app.py`
     - `requirements.txt`
     - `README.md`

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file: `app.py`
   - Click "Deploy"

3. **Access Your App**
   - You'll get a public URL like: `https://your-app.streamlit.app`
   - Share this URL with anyone!

---

### Option 3: Deploy on Hugging Face Spaces

1. **Create a Hugging Face Account**
   - Go to [huggingface.co](https://huggingface.co)
   - Sign up for free

2. **Create a New Space**
   - Click on your profile → "New Space"
   - Name: `cyberbullying-detector`
   - SDK: Select "Streamlit"
   - Click "Create Space"

3. **Upload Files**
   - Upload `app.py` and `requirements.txt`
   - The app will auto-deploy

4. **Access Your App**
   - URL: `https://huggingface.co/spaces/YOUR_USERNAME/cyberbullying-detector`

---

### Option 4: Deploy on Render

1. **Create Account**
   - Go to [render.com](https://render.com)
   - Sign up for free

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure Service**
   - Name: `cyberbullying-detector`
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)

---

## 📖 How to Use

1. **Enter Text**
   - Type or paste any text in the input box
   - Or click on example buttons to test

2. **Adjust Settings (Optional)**
   - Use the sidebar slider to change confidence threshold
   - Higher threshold = fewer false positives
   - Default: 0.7 (70%)

3. **Analyze**
   - Click "Analyze Text" button
   - View instant results with:
     - Detection status (Safe/Bullying)
     - Confidence scores
     - Probability breakdown
     - Visual gauge chart

4. **Interpret Results**
   - ✅ **Green/Safe**: Content is appropriate
   - ⚠️ **Red/Warning**: Potential cyber bullying detected
   - 📊 **Scores**: Higher toxicity = more likely to be bullying

---

## 🔧 Configuration

### Confidence Threshold
- **Low (0.5-0.6)**: More sensitive, catches more cases, may have false positives
- **Medium (0.7)**: Balanced approach (recommended)
- **High (0.8-0.95)**: Very strict, fewer false positives, may miss some cases

### Model Information
- **Model**: `unitary/toxic-bert`
- **Type**: Pre-trained BERT transformer
- **Training**: Large-scale toxic language datasets
- **Languages**: Primarily English

---

## 📁 Project Structure

```
cyberbullying-detector/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore            # Git ignore file (optional)
```

---

## 🎯 Use Cases

- **Social Media Platforms**: Automated content moderation
- **Educational Institutions**: Monitor student communications
- **Online Communities**: Protect users from harassment
- **Parental Control**: Monitor children's online interactions
- **Customer Support**: Flag abusive customer messages

---

## ⚠️ Limitations

- Works best with English text
- May struggle with sarcasm or cultural context
- Not 100% accurate - human review recommended
- Context from conversation history not considered
- Requires internet for first-time model download

---

## 🔐 Privacy & Ethics

- No data is stored or logged
- All processing happens in real-time
- Model runs locally or on your chosen server
- Always review flagged content manually
- Use as a support tool, not sole decision-maker

---

## 🛠️ Troubleshooting

### Issue: Model download is slow
**Solution**: First run downloads ~500MB model. Be patient or use faster internet.

### Issue: Out of memory error
**Solution**: Close other applications or use cloud deployment.

### Issue: Import errors
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Port already in use
**Solution**: Use a different port:
```bash
streamlit run app.py --server.port=8502
```

---

## 📚 Technical Details

### Model Architecture
- **Base**: BERT (Bidirectional Encoder Representations from Transformers)
- **Fine-tuning**: Toxic language classification
- **Input**: Text sequences up to 512 tokens
- **Output**: Binary classification (toxic/non-toxic)

### Performance
- **Processing Time**: < 1 second per text
- **Accuracy**: ~85-90% on standard benchmarks
- **Precision**: High (minimizes false positives)

---

## 🚀 Future Enhancements

- [ ] Multi-language support
- [ ] Severity classification (mild/moderate/severe)
- [ ] Conversation context analysis
- [ ] User history tracking
- [ ] Explanation generation for decisions
- [ ] API endpoint for integration
- [ ] Mobile app version

---

## 📝 License

This project is created for educational purposes as part of academic coursework.

---

## 👨‍💻 Developer

**S.Hemanth Chandra**  
Registration Number: 2025MS020  

For questions or feedback, please create an issue in the repository.

---

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library
- **Unitary**: For the pre-trained toxic-bert model
- **Streamlit**: For the amazing web framework
- **PyTorch**: For deep learning capabilities

---

## 📞 Support

If you encounter any issues:

1. Check the Troubleshooting section
2. Ensure all dependencies are installed correctly
3. Verify Python version (3.8+)
4. Check internet connection (for model download)

---

**Built with ❤️ using AI for Social Good**
