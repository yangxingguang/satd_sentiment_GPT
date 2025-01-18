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


## Code Explanation
### 1.Fine-tuning GPT  

### 1.1 Setting up an OpenAI API key

```
openai.api_key = ""  
```

Before using the OpenAI API, you first need to set up an API key. You can get your own API key on the OpenAI developer platform and replace it with the corresponding location in the code.

#### 1.2 Loading review and sentiment label data
```
def load_labels(file_path):
    comments = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            for message in data['messages']:
                if message['role'] == 'user':
                    comments.append(message['content'])  
                elif message['role'] == 'assistant':
                    labels.append(message['content'])  
    return comments, labels
```
This part of the function is to load data from the specified file. Each line of data is in JSON format and contains the messages field, where role indicates the source of the message (user or assistant) and content is the comment or sentiment label. This function extracts user comments and sentiment labels into the comments and labels lists respectively, and returns these two lists.

#### 1.3 Sentiment prediction using the fine-tuned GPT model
```
def predict_sentiment_openai(comment, model_id):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[{"role": "user", "content": f"Perform sentiment analysis on the following comment: '{comment}'"}],
        max_tokens=200, n=1, stop=None, temperature=0.5,
    )
    sentiment = response.choices[0].message.content.strip()
    return sentiment
```
In this code, we use the fine-tuned GPT model for sentiment prediction. Through the openai.ChatCompletion.create method, we pass the user comment to the model and get the sentiment prediction. model_id is the ID of the fine-tuned model, and the prediction result returned can be 'negative' or 'non-negative'.


#### 1.4 Loading test data and performing sentiment prediction
```
file_path = ""  
test_comments, true_labels = load_labels(file_path)
```
The test data is loaded by calling the load_labels() function, which returns the comment list test_comments and the true labels true_labels.
```
model_id = ""  
predictions_openai = [predict_sentiment_openai(comment, model_id) for comment in test_comments]
```
Use the fine-tuned GPT model to predict the sentiment of each review and save the results in the predictions_openai list.

#### 1.5 Print the model's predictions
```
print("Model predictions:", predictions_openai)
```
Output the sentiment prediction results of the model to help check the prediction effect of the model.
#### 1.6 Mapping labels to values
```
label_mapping = {
    'non-negative': 1,
    'negative': 0
}
```
To calculate evaluation metrics such as precision, recall, and F1 score, we map sentiment labels such as 'non-negative' and 'negative' to numeric values such as 1 and 0.

#### 1.7 Convert true labels and predictions to numerical values
```
true_labels_numeric = [label_mapping[label] for label in true_labels]
```
Map true labels from strings (such as 'negative') to numeric values (such as 0), generating the true_labels_numeric list.
```
predictions_numeric = []for pred in predictions_openai:
    if pred in label_mapping:
        predictions_numeric.append(label_mapping[pred])
    else:
        predictions_numeric.append(0)  
```
Convert the model's predicted labels to numeric values, using a default value of 0 if the predicted label is not in the mapping.

#### 1.8 Calculating evaluation metrics
```
precision = precision_score(true_labels_numeric, predictions_numeric, average='weighted')
recall = recall_score(true_labels_numeric, predictions_numeric, average='weighted')
f1 = f1_score(true_labels_numeric, predictions_numeric, average='weighted')
```
Calculate the model's evaluation metrics: Precision, Recall, and F1 score.

#### 1.9 Output evaluation results
```
print(f"Weighted Precision: {precision:.3f}")print(f"Weighted Recall: {recall:.3f}")print(f"Weighted F1 Score: {f1:.3f}")
```
Output precision, recall, and F1 score to evaluate the sentiment analysis performance of the model.

### 2. Deep Learning

#### 2.1 Import libraries and initialize configuration
First, we imported the required Python libraries and did some necessary initial setup:
```
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import GlobalAveragePooling1D
```
These libraries help us with data processing (such as pandas and numpy), implement machine learning models (such as tensorflow), and calculate performance evaluation metrics of the models (such as precision_score, recall_score, f1_score).

#### 2.2 File path and initialization configuration
```
train_file_paths = [f'without{i+1}.csv' for i in range(10)]  
test_file_paths = [f's{i+1}_subset.csv' for i in range(10)]  
tokenizer = Tokenizer(num_words=5000)
max_len = 100
```
Here, we define the file paths for the training and test datasets and initialize the Tokenizer object to convert text data into numerical form. max_len sets the maximum length of the input sequence to ensure that the dimensions of the input data are consistent.

We also define a results_metrics dictionary to store the performance evaluation metrics (precision, recall, and F1 score) for each model.
```
results_metrics = {
    "CNN": {"Precision": [], "Recall": [], "F1": []},
    "LSTM": {"Precision": [], "Recall": [], "F1": []},
    "Transformer": {"Precision": [], "Recall": [], "F1": []},
}
```

#### 2.3 Cross-validation: Model training and evaluation
We use 10-fold cross validation to train and evaluate three different models: CNN, LSTM, and Transformer. In each iteration, we load training data and test data, and tokenize and pad the text through Tokenizer to ensure that the data format input to the model is consistent.
```
for fold_num in range(10):
    print(f"Fold {fold_num + 1}:")
    train_data = pd.read_csv(train_file_paths[fold_num])  
    test_data = pd.read_csv(test_file_paths[fold_num])  
    tokenizer.fit_on_texts(train_data['Comment'])  
 Tokenizer
```
Here, we load the training and test data for the current fold, tokenize the text data using Tokenizer, and pad the text sequences to the same length (max_len) to fit the model.
```
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data['Comment']), maxlen=max_len)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['Comment']), maxlen=max_len)
```

#### 2.4 Label processing
Convert sentiment labels (such as negative sentiment) to numerical form:
```
y_train = to_categorical((train_data['Sentiment'] == 'negative').astype(int))
y_test = to_categorical((test_data['Sentiment'] == 'negative').astype(int))
```
The labels are converted to one-hot encoding using the to_categorical method, where negative sentiment ('negative') is labeled as 1 and other sentiments are labeled as 0.

#### 2.5 CNN Model
```
cnn_input = Input(shape=(max_len,))
cnn_embedding = Embedding(input_dim=5000, output_dim=128)(cnn_input)
filter_sizes = [2, 3, 4]
cnn_layers = [GlobalMaxPooling1D()(Conv1D(filters=128, kernel_size=fs, activation='relu')(cnn_embedding)) for fs in filter_sizes]
cnn_concat = tf.keras.layers.Concatenate()(cnn_layers)
cnn_dense = Dense(128, activation='relu')(cnn_concat)
cnn_dropout = Dropout(0.5)(cnn_dense)
cnn_output = Dense(2, activation='softmax')(cnn_dropout)
cnn_model = Model(inputs=cnn_input, outputs=cnn_output)
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
```
In the CNN model, we first convert the text into a vector representation through the embedding layer, and then apply multiple convolutional layers (Conv1D), each of which uses filters of different sizes to extract different features. After that, the maximum value in the feature map is selected through the GlobalMaxPooling1D layer and passed to the fully connected layer and the Dropout layer to prevent overfitting. The prediction output by the model finally calculates the classification probability through the softmax activation function.

```
y_pred_cnn = cnn_model.predict(X_test).argmax(axis=1)
y_true_cnn = y_test.argmax(axis=1)
precision_cnn = precision_score(y_true_cnn, y_pred_cnn, average='weighted', zero_division=0)
recall_cnn = recall_score(y_true_cnn, y_pred_cnn, average='weighted', zero_division=0)
f1_cnn = f1_score(y_true_cnn, y_pred_cnn, average='weighted', zero_division=0)
results_metrics["CNN"]["Precision"].append(precision_cnn)
results_metrics["CNN"]["Recall"].append(recall_cnn)
results_metrics["CNN"]["F1"].append(f1_cnn)
```
We use the test data to evaluate the precision, recall, and F1 score of the CNN model and store the results in the results_metrics dictionary.

#### 2.6 LSTM Model
```
lstm_input = Input(shape=(max_len,))
lstm_embedding = Embedding(input_dim=5000, output_dim=128)(lstm_input)
lstm_layer = Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2))(lstm_embedding)
lstm_dense = Dense(128, activation='relu')(lstm_layer)
lstm_dropout = Dropout(0.5)(lstm_dense)
lstm_output = Dense(2, activation='softmax')(lstm_dropout)
lstm_model = Model(inputs=lstm_input, outputs=lstm_output)
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
```
The LSTM model uses bidirectional LSTM (Bidirectional(LSTM)), which can capture both past and future contextual information of the input sequence. Similar to the CNN model, it uses an embedding layer, a fully connected layer, and a Dropout layer to improve the generalization ability of the model.

```
y_pred_lstm = lstm_model.predict(X_test).argmax(axis=1)
y_true_lstm = y_test.argmax(axis=1)
precision_lstm = precision_score(y_true_lstm, y_pred_lstm, average='weighted', zero_division=0)
recall_lstm = recall_score(y_true_lstm, y_pred_lstm, average='weighted', zero_division=0)
f1_lstm = f1_score(y_true_lstm, y_pred_lstm, average='weighted', zero_division=0)
results_metrics["LSTM"]["Precision"].append(precision_lstm)
results_metrics["LSTM"]["Recall"].append(recall_lstm)
results_metrics["LSTM"]["F1"].append(f1_lstm)
```
Similar to the CNN model, the LSTM model is evaluated for precision, recall, and F1 score and the results are stored in the results_metrics dictionary.

#### 2.7 Transformer Model
```
transformer_input = Input(shape=(max_len,))
transformer_embedding = Embedding(input_dim=5000, output_dim=120)(transformer_input)
transformer_pool = GlobalAveragePooling1D()(transformer_embedding)
transformer_dense1 = Dense(1024, activation='relu')(transformer_pool)
transformer_dropout = Dropout(0.5)(transformer_dense1)
transformer_dense2 = Dense(1024, activation='relu')(transformer_dropout)
transformer_output = Dense(2, activation='softmax')(transformer_dense2)
transformer_model = Model(inputs=transformer_input, outputs=transformer_output)
transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
transformer_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
```
The Transformer model uses the GlobalAveragePooling1D layer to pool the features of the entire sequence and performs classification through multiple fully connected layers. The model output is used for sentiment classification through the softmax activation function.

```
y_pred_transformer = transformer_model.predict(X_test).argmax(axis=1)
y_true_transformer = y_test.argmax(axis=1)
precision_transformer = precision_score(y_true_transformer, y_pred_transformer, average='weighted', zero_division=0)
recall_transformer = recall_score(y_true_transformer, y_pred_transformer, average='weighted', zero_division=0)
f1_transformer = f1_score(y_true_transformer, y_pred_transformer, average='weighted', zero_division=0)
results_metrics["Transformer"]["Precision"].append(precision_transformer)
results_metrics["Transformer"]["Recall"].append(recall_transformer)
results_metrics["Transformer"]["F1"].append(f1_transformer)
```
Likewise, the evaluation results of the Transformer model are stored in the results_metrics dictionary.

#### 2.8 Print the final evaluation results
```
for model_name, metrics in results_metrics.items():
    print(f"\n{model_name}:")
    for i in range(10):
        print(f"  Fold {i + 1}: Precision={metrics['Precision'][i]:.3f}, Recall={metrics['Recall'][i]:.3f}, F1={metrics['F1'][i]:.3f}")
    print(f"  Average Precision: {np.mean(metrics['Precision']):.3f}")
    print(f"  Average Recall: {np.mean(metrics['Recall']):.3f}")
    print(f"  Average F1: {np.mean(metrics['F1']):.3f}")
```
Finally, we output the evaluation results of each model in 10-fold cross validation and calculate and print the average precision, recall, and F1 score for each model.

### 3. Traditional Machine Learning

#### 3.1 Import libraries and initialize configuration
```
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
```
We imported libraries for data processing, model training, and performance evaluation:
- pandas  Used to load and process data.
- TfidfVectorizer  Used to convert text data into numeric representation.
- LabelEncoder  Used to encode labels (sentiment labels) into numerical form.
- MultinomialNB, LogisticRegression, RandomForestClassifier, SVC, KNeighborsClassifier  Used to implement different traditional machine learning models.
- precision_score, recall_score, f1_score  Used to evaluate the performance of the model.

#### 3.2 File path and initialization configuration
```
train_file_paths = [f'without{i+1}.csv' for i in range(10)]  
test_file_paths = [f's{i+1}_subset.csv' for i in range(10)]  
label_encoder = LabelEncoder()
```
We define the file paths for the training and testing datasets and initialize a LabelEncoder to encode the sentiment labels.

```
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', random_state=42, probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}
```
Five traditional machine learning models are defined here: Naive Bayes, Logistic Regression, Random Forest, Support Vector Machine (SVM), and K Nearest Neighbors (KNN).

#### 3.3 Initialize model evaluation metrics
```
model_metrics = {model_name: {"Precision": [], "Recall": [], "F1": []} for model_name in models.keys()}
```
We initialize a dictionary model_metrics to store the precision, recall, and F1 score of each model.

#### 3.4 Cross-validation: training and evaluating each model
```
for i in range(10):
    train_data = pd.read_csv(train_file_paths[i])  
    test_data = pd.read_csv(test_file_paths[i])   
```
In 10-fold cross validation, we load the training data and test data for the current fold.

```
y_train = label_encoder.fit_transform(train_data['Sentiment'])  
X_train = train_data['Comment']  
y_test = label_encoder.transform(test_data['Sentiment'])  
X_test = test_data['Comment']  
```
We use LabelEncoder to convert sentiment labels into numerical form for training machine learning models.

```
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  
X_test_tfidf = tfidf_vectorizer.transform(X_test)  
```
We use TfidfVectorizer to convert text reviews into TF-IDF feature representations, using 1-gram and 2-gram (i.e., words and bigrams) to represent the features of the text.

```
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)  
    y_pred = model.predict(X_test_tfidf)  
```
Each model is trained and predicted on the test data.

```
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)  
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)  
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)  
```
We compute the precision, recall, and F1 score for each model.

```
model_metrics[model_name]["Precision"].append(precision)
model_metrics[model_name]["Recall"].append(recall)
model_metrics[model_name]["F1"].append(f1)
```
Store the calculated metrics in the model_metrics dictionary.

#### 3.5 Output the evaluation results of each model
```
for model_name, metrics in model_metrics.items():
    print(f"\n{model_name}:")
    for fold in range(10):
        print(f"  Fold {fold + 1}: Precision={metrics['Precision'][fold]:.3f}, Recall={metrics['Recall'][fold]:.3f}, F1={metrics['F1'][fold]:.3f}")
    print(f"  Average Precision: {sum(metrics['Precision']) / 10:.3f}")
    print(f"  Average Recall: {sum(metrics['Recall']) / 10:.3f}")
    print(f"  Average F1 Score: {sum(metrics['F1']) / 10:.3f}")
```
Print the precision, recall, and F1 score of each model in 10-fold cross validation, as well as the average evaluation metric for each model.

## Experimental procedures
### 1.Install required libraries
First, make sure you have Python 3.7 or higher installed. Then, install the following required libraries via pip:
```
pip install openai pandas numpy tensorflow scikit-learn
```
### 2.Preparing data for fine-tuning
Make sure the data has been converted to JSONL format suitable for OpenAI fine-tuning. Each record should contain a messages field, which is a list containing multiple elements. Each element is an object with the role fields user (representing the user's comment), assistant (representing the assistant's sentiment label), and system (representing the description of the sentiment analysis task). The following is an example:
```
{
  "messages": [
    {"role": "system", "content": "Perform sentiment analysis."},
    {"role": "user", "content": "// FIXME: Is \"No Namespace is Empty Namespace\" really OK?"},
    {"role": "assistant", "content": "negative"}
  ]}
```
In this example, messages with the role of system are used to indicate the task type (such as sentiment analysis), messages with the role of user contain user comments, and messages with the role of assistant contain sentiment labels (such as "negative").

### 3.Fine-tuning the GPT model
The fine-tuning process will be performed on OpenAI's API, and you do not need to train the model locally. Here are the specific steps:
1. Log in to the openai official website, click products and select the API login interface;

2. Click the Dashboard interface, find Fine-tuning on the left side of the page to enter the fine-tuning interface, and click create to start the fine-tuning job;

3. Upload the processed data;

4. Select the appropriate fine-tuning model and start fine-tuning;

5. After completion, get the fine-tuned model ID;

After completing fine-tuning, you will get a model ID, which is used for subsequent model calls.

### 4. Evaluating the fine-tuned GPT model
In this step, you will evaluate the fine-tuned GPT model, manually evaluating it using the corresponding s_fold_i.jsonl test data (where i represents the number of the fold) and the model ID obtained in the previous step. You will evaluate each fold (s_fold_1.jsonl to s_fold_10.jsonl) and calculate the precision, recall, and F1 score for that fold. This process will be repeated 10 times, using a different fold for evaluation each time.

### 5. Running the deep learning model

Run the deep learning model script to train sentiment analysis models (including CNN, LSTM, and Transformer). The script will process text data and train the model, outputting evaluation metrics of the model on the test set (such as precision, recall, and F1 score).

### 6. Running Traditional Machine Learning Models

Run the traditional machine learning model script to train a sentiment analysis model. This script also processes text data and trains the model, outputting evaluation metrics of the model on the test set (such as precision, recall, and F1 score).

### 7. Comparison Models

The performance of each model on the test data is evaluated through 10-fold cross validation. The performance of each model is evaluated by precision, recall, and F1 score, and the results are printed in the terminal output.
