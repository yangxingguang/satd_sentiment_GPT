# Fine-tuning GPT Model for SATD Sentiment Analysis

## Project Overview
This project aims to explore the effectiveness of fine-tuning the GPT model for the Self-Reported Technical Debt (SATD) sentiment analysis task. By comparing the performance of traditional machine learning models, deep learning models, and the fine-tuned GPT model, the project seeks to validate the advantages of the GPT model in capturing emotional features and modeling complex contexts. The experimental design includes various models and parameter configurations to ensure the comprehensiveness and reliability of the results. The research addresses the following key questions:

- **RQ1**: How does the performance of the fine-tuned GPT model compare to traditional machine learning methods in the SATD sentiment analysis task?  
  The fine-tuned GPT model captures deep semantic information through pretraining, which may provide significant advantages in the comprehensiveness and accuracy of sentiment analysis compared to traditional methods (such as SVM, Random Forest, etc.) that rely on shallow features.

- **RQ2**: How does the fine-tuned GPT model perform in sentiment capture compared to other deep learning models?  
  With its self-attention mechanism, the GPT model may outperform other deep learning models (such as CNN, LSTM, and Transformer) in understanding complex contexts and capturing multi-dimensional emotions.

## Prerequisites
- **Python 3.7 or higher**
- **Required Libraries**: The project uses multiple libraries and frameworks for data processing, model training, and evaluation, including:
  - `numpy` and `pandas`: for data handling and manipulation.
  - `scikit-learn`: for traditional machine learning tasks, including `TfidfVectorizer` (for text vectorization), `LabelEncoder` (for sentiment label encoding), and common classification algorithms (e.g., `LogisticRegression`, `RandomForestClassifier`).
  - `tensorflow`: for deep learning model construction, including CNN, LSTM, etc.
  - `openai`: for fine-tuning the GPT model.
  - `json`: for processing JSON data, especially for converting data to the JSONL format for GPT fine-tuning.

## Tools and Environment
- **IDE/Code Editor**: It is recommended to use Visual Studio Code, PyCharm, or any other Python-compatible IDE for development and debugging.
- **Jupyter Notebook**: An optional tool, suitable for interactive programming and result visualization.
- **Operating System**: The project supports cross-platform use and is compatible with Windows, macOS, and Linux.

## Data Files
The SATD dataset used in this project has been preprocessed, keeping only the `Comment` and `Sentiment` fields and removing samples labeled as `mixed`, `exclude`, and `no agreement`. The dataset has been manually divided into 10 folds for cross-validation. Each fold contains both training and testing data, with the following file structure:

- **Training Data**: The training data for each fold is stored in the file `withouti.csv` (where `i` is the fold number, i = 1 to 10). These files contain comments and sentiment data after removing invalid labels.
- **Testing Data**: The testing data for each fold is stored in the file `si_subset.csv`, which corresponds to the respective training data file.
- **GPT Fine-tuning Data**: The training data has been converted into the JSONL format for GPT fine-tuning. The training data for each fold is stored as `withouti.jsonl`, and the corresponding testing data is stored as `s_fold_i.jsonl`, which is used to validate the fine-tuned model.

## Program Description
This project consists of three main parts, each responsible for different tasks:

- **Deep Learning.py**: Implements the deep learning portion, constructing and training CNN, LSTM, and Transformer models for sentiment analysis. It also performs 10-fold cross-validation for training and evaluation.
- **Machine Learning.py**: Implements the traditional machine learning portion, using algorithms such as logistic regression, random forest, and support vector machines for sentiment analysis.
- **Fine-tuning GPT.py**: After fine-tuning, the corresponding `s_fold_i.jsonl` files are used to test and validate the fine-tuned model.

## Usage
1. Place all project files in the same directory.
2. Run the corresponding model script:
   - For traditional machine learning or deep learning models, run the relevant Python script.
   - For the GPT fine-tuning portion, users need to manually upload the `withouti.csv` file to OpenAI's API for fine-tuning. Once fine-tuning is complete, use the corresponding `s_fold_i.jsonl` test data to evaluate the fine-tuned model.
