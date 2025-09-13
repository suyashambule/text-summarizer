# Text Summarization with Transformers

An end-to-end text summarization system built with Transformers and Pegasus model that can automatically generate concise summaries from dialogues and conversations. This project uses the SAMSum dataset for training and provides a complete ML pipeline from data ingestion to model evaluation.

## 📚 About the Project

This text summarization project leverages the power of Google's Pegasus model to create accurate and meaningful summaries of conversational text. The system is designed with a modular architecture that includes data ingestion, preprocessing, model training, and evaluation components.

## ✨ Features

- **State-of-the-art Model**: Uses Google's Pegasus model pre-trained on CNN/DailyMail
- **End-to-End Pipeline**: Complete ML workflow from data ingestion to evaluation
- **Conversational Focus**: Specialized in summarizing dialogues and conversations
- **Modular Architecture**: Well-structured codebase with separate components
- **Research Notebooks**: Comprehensive Jupyter notebooks for experimentation
- **Docker Support**: Containerized deployment ready
- **Evaluation Metrics**: ROUGE scores and BLEU scores for performance assessment
- **Logging System**: Comprehensive logging for monitoring and debugging

## 🎯 Use Cases

- **Chat Summary**: Summarize long chat conversations
- **Meeting Notes**: Generate concise meeting summaries
- **Customer Support**: Summarize customer support interactions
- **Social Media**: Create brief summaries of social media conversations
- **Email Threads**: Condense long email discussions

## 🛠️ Project Structure

```
text-summarizer/
│
├── src/textsummarizer/          # Main source code
│   ├── components/              # Pipeline components
│   │   └── data_ingestion.py   # Data downloading and extraction
│   ├── config/                 # Configuration management
│   ├── entity/                 # Data classes and entities
│   ├── pipeline/               # ML pipeline stages
│   ├── utils/                  # Utility functions
│   └── logging/                # Logging configuration
│
├── config/                      # Configuration files
│   └── config.yaml             # Main configuration
├── research/                    # Jupyter notebooks
│   ├── 1_data_ingestion.ipynb  # Data loading experiments
│   ├── 2_data_transformation.ipynb # Data preprocessing
│   └── 3_model_trainer.ipynb   # Model training experiments
├── artifacts/                   # Generated artifacts
│   └── data_ingestion/         # Downloaded and processed data
├── logs/                       # Application logs
│
├── app.py                      # Web application (if implemented)
├── main.py                     # Pipeline execution script
├── requirements.txt            # Python dependencies
├── params.yaml                 # Training parameters
├── Dockerfile                  # Docker configuration
└── setup.py                    # Package setup
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers library
- 8GB+ RAM recommended for training
- GPU recommended for faster training

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd text-summarizer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the pipeline**
   ```bash
   python main.py
   ```

### Docker Setup

1. **Build the Docker image**
   ```bash
   docker build -t text-summarizer .
   ```

2. **Run the container**
   ```bash
   docker run text-summarizer
   ```

## 📊 Dataset

### SAMSum Dataset

The project uses the SAMSum dataset, which contains:

- **Training samples**: ~14,000 conversations with summaries
- **Validation samples**: ~800 conversations
- **Test samples**: ~800 conversations
- **Format**: CSV files with dialogue and summary columns
- **Domain**: Messenger-like conversations

**Example:**
```
Dialogue: "Amanda: I baked cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)"

Summary: "Amanda baked cookies and will bring Jerry some tomorrow."
```

## 🔧 Model Architecture

### Pegasus Model

- **Base Model**: `google/pegasus-cnn_dailymail`
- **Architecture**: Transformer-based encoder-decoder
- **Pre-training**: CNN/DailyMail dataset
- **Fine-tuning**: SAMSum conversational data
- **Tokenizer**: SentencePiece-based tokenization

### Training Configuration

```yaml
Training Parameters:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  warmup_steps: 500
  weight_decay: 0.01
  learning_rate: 5e-5
  evaluation_strategy: steps
  eval_steps: 500
```

## 📈 Pipeline Stages

### 1. Data Ingestion
- Downloads SAMSum dataset from remote source
- Extracts and organizes data files
- Validates data integrity

### 2. Data Transformation
- Tokenizes dialogues and summaries
- Handles sequence length limitations
- Prepares data for model training

### 3. Model Training
- Fine-tunes Pegasus model on SAMSum data
- Implements gradient accumulation for larger effective batch size
- Saves model checkpoints

### 4. Model Evaluation
- Calculates ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- Generates sample predictions
- Saves evaluation metrics

## 💻 Usage

### Running the Full Pipeline

```bash
python main.py
```

This executes all pipeline stages:
1. Data ingestion
2. Data transformation
3. Model training
4. Model evaluation

### Research and Experimentation

Explore the research notebooks for detailed analysis:

```bash
jupyter notebook research/
```

- `1_data_ingestion.ipynb`: Data loading and exploration
- `2_data_transformation.ipynb`: Text preprocessing experiments
- `3_model_trainer.ipynb`: Model training and fine-tuning

### Custom Summarization

```python
from transformers import pipeline

# Load the trained model
summarizer = pipeline("summarization", 
                     model="artifacts/model_trainer/pegasus-samsum-model")

# Summarize text
text = "Your long conversation here..."
summary = summarizer(text, max_length=50, min_length=10)
print(summary[0]['summary_text'])
```

## 📊 Evaluation Metrics

The model is evaluated using:

- **ROUGE-1**: Unigram overlap between generated and reference summaries
- **ROUGE-2**: Bigram overlap for better semantic evaluation
- **ROUGE-L**: Longest common subsequence for structural similarity
- **BLEU Score**: Precision-based evaluation metric

## 🔍 Configuration

### config.yaml
```yaml
data_ingestion:
  source_URL: https://github.com/krishnaik06/datasets/raw/refs/heads/main/summarizer-data.zip
  local_data_file: artifacts/data_ingestion/data.zip
  
model_trainer:
  model_ckpt: google/pegasus-cnn_dailymail
  
model_evaluation:
  metric_file_name: artifacts/model_evaluation/metrics.csv
```

### params.yaml
```yaml
TrainingArguments:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  warmup_steps: 500
  weight_decay: 0.01
```

## 🚀 Future Enhancements

- [ ] Web interface for interactive summarization
- [ ] REST API for integration with other applications
- [ ] Support for multiple languages
- [ ] Real-time summarization capabilities
- [ ] Integration with cloud platforms (AWS, GCP)
- [ ] Advanced evaluation metrics
- [ ] Model quantization for faster inference
- [ ] Batch processing capabilities

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Reduce `per_device_train_batch_size` in params.yaml
- Increase `gradient_accumulation_steps`
- Use CPU training if necessary

**Download Errors**:
- Check internet connection
- Verify dataset URL in config.yaml
- Clear artifacts folder and retry

**Model Loading Issues**:
- Ensure all dependencies are installed
- Check model path in configuration
- Verify PyTorch and Transformers versions

## 📋 Requirements

```
transformers>=4.20.0
datasets>=2.0.0
torch>=1.12.0
rouge_score>=0.1.0
sacrebleu>=2.0.0
pandas>=1.4.0
nltk>=3.7
tqdm>=4.64.0
PyYAML>=6.0
fastapi>=0.78.0
uvicorn>=0.18.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes and add tests
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## 📚 Technologies Used

- **🤗 Transformers**: State-of-the-art NLP models
- **PyTorch**: Deep learning framework
- **Datasets**: Efficient data loading and processing
- **NLTK**: Natural language processing utilities
- **FastAPI**: Modern web framework for APIs
- **Docker**: Containerization platform
- **YAML**: Configuration management
- **Pandas**: Data manipulation and analysis

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Research**: For the Pegasus model
- **Hugging Face**: For the Transformers library
- **Samsung**: For the SAMSum dataset
- **Open Source Community**: For the amazing tools and libraries

## 📞 Contact

For questions, suggestions, or issues, please open an issue in the repository.

---

*Transform long conversations into concise summaries with the power of AI! 🚀📝*
