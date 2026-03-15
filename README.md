# mini-project-9

# Mini Project IX: Content Moderation with Transformers

## Problem Description and Motivation

Content moderation is a critical challenge in today's digital landscape, where social media platforms and online communities must balance free expression with the prevention of harmful content. This project focuses on automating content moderation using machine learning, specifically targeting the detection of hate speech and offensive language in text data. By leveraging advanced transformer models, we aim to develop a robust system that can classify tweets into categories: Hate Speech, Offensive, or Neither. This work is motivated by the need for scalable, accurate, and efficient moderation tools to foster safer online environments, reducing the burden on human moderators and enabling real-time intervention.

## Dataset Description

The dataset used is the "Hate Speech and Offensive Language" dataset, sourced from a public GitHub repository. It contains labeled tweets categorized into three classes: Hate Speech (0), Offensive (1), and Neither (2). The dataset consists of approximately 24,000 samples, with an imbalanced distribution where "Offensive" is the majority class. The data includes raw tweet text, which has been preprocessed in this project to remove URLs, normalize mentions, and strip non-ASCII characters.

Source: [Hate Speech and Offensive Language Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language)

## Setup Instructions and How to Run

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git (for cloning the repository)

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/Ledja22/mini-project-9.git
   cd mini-project-9
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Project
The project is organized into three Jupyter notebooks, to be run in sequence:

1. **01_exploration.ipynb**: Data loading, exploration, preprocessing, and train/val/test splitting. Saves processed data to `data/npy_files/`.
2. **02_baseline.ipynb**: Trains and evaluates a TF-IDF + Logistic Regression baseline model. Saves metrics and plots to `data/content/`.
3. **03_transformer.ipynb**: Fine-tunes DistilBERT, evaluates performance, and performs error analysis. Saves metrics, plots, and model weights.

Run the notebooks in order using Jupyter Notebook:
```
jupyter notebook notebooks/01_exploration.ipynb
```
Then proceed to the next notebooks. Ensure the `data/` directory structure exists (create `data/content/` and `data/npy_files/` if needed).

## Results Summary: Baseline vs Transformer Comparison

| Metric          | Model              | Hate Speech | Offensive | Neither | Macro Avg | Weighted Avg |
|-----------------|--------------------|-------------|-----------|---------|-----------|--------------|
| Precision      | TF-IDF + LR       | 0.50        | 0.92      | 0.90    | 0.77      | 0.90         |
|                 | DistilBERT        | 0.65        | 0.95      | 0.92    | 0.84      | 0.93         |
| Recall         | TF-IDF + LR       | 0.45        | 0.91      | 0.91    | 0.76      | 0.89         |
|                 | DistilBERT        | 0.60        | 0.94      | 0.93    | 0.82      | 0.92         |
| F1-Score       | TF-IDF + LR       | 0.47        | 0.91      | 0.90    | 0.76      | 0.89         |
|                 | DistilBERT        | 0.62        | 0.94      | 0.92    | 0.83      | 0.92         |

The transformer model (DistilBERT) outperforms the baseline in all metrics, particularly for the minority "Hate Speech" class, demonstrating the value of contextual embeddings for nuanced text classification.

## Team Member Contributions
