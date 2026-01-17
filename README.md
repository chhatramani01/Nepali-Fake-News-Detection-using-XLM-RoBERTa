
# 🚀 Fine-tuning XLM-RoBERTa for Nepali Fake News Detection

This project follows a systematic approach to fine-tune a multilingual Transformer model for a specific low-resource language task.

## **Step 1: Environment Setup & Installations**

The first step involves setting up the environment. The project is designed to run on a **GPU-enabled environment** (like Google Colab's T4 GPU). You must install the necessary Hugging Face and machine learning libraries:

* `transformers`: For model architecture and training.  
* `datasets`: To handle and process the news data.  
* `evaluate` & `scikit-learn`: For calculating metrics like Accuracy and F1 Score.  

## **Step 2: Model Selection & Tokenization**

The model chosen is **`xlm-roberta-base`**, a powerful multilingual model pre-trained on 100 languages, including Nepali.

* **Tokenizer Initialization:** We load the `AutoTokenizer` corresponding to the XLM-RoBERTa checkpoint.  
* **Functionality:** The tokenizer converts raw Nepali text into numerical "tokens" and "input IDs" that the transformer can understand.  

## **Step 3: Data Preprocessing**

Before feeding data into the model, it must be cleaned and formatted:

1. **Loading:** The dataset containing Nepali news articles labeled as "Fake" or "Real" is loaded.  
2. **Mapping Labels:** Text labels are mapped to integers (e.g., Fake → 0, Real → 1).  
3. **Tokenization Mapping:** A preprocessing function is applied across the entire dataset to truncate long articles and pad shorter ones to a uniform length.  

## **Step 4: Model Configuration**

We initialize the model for sequence classification using `AutoModelForSequenceClassification`:

* The model is loaded with a classification head specifically for two classes (Real/Fake).  
* It is automatically moved to the **GPU (CUDA)** to accelerate the training process.  

## **Step 5: Defining Training Hyperparameters**

The fine-tuning process is governed by specific parameters defined in `TrainingArguments`:

* **Learning Rate:** A small learning rate (e.g., 2e-5) is used to avoid overwriting the pre-trained knowledge too quickly.  
* **Epochs:** The number of times the model sees the entire dataset.  
* **Evaluation Strategy:** The model is evaluated at the end of every epoch to monitor its performance on unseen data.  
* **Logging:** Training and validation loss are logged to track the "health" of the training process.  

## **Step 6: Training (Fine-tuning)**

The `Trainer` API is used to execute the training loop. During this stage:

1. **Forward Pass:** The model makes predictions on a batch of Nepali news articles.  
2. **Loss Calculation:** The difference between the prediction and the actual label is calculated.  
3. **Backward Pass:** The model's weights are adjusted to minimize the loss.  

## **Step 7: Evaluation & Metrics**

After training, the model's effectiveness is measured using the validation set. Key metrics recorded in this project include:

* **Accuracy:** Overall correctness.  
* **F1 Score:** Crucial for fake news, as it balances the identification of both Real and Fake news without bias.  
* **Confusion Matrix:** Used to see where the model is making specific mistakes (e.g., misclassifying "Real" as "Fake").  

## **Step 8: Results & Conclusion**

The final fine-tuned model achieved an impressive **99.49% Accuracy and F1 Score** on the test dataset.

* **Healthy Behavior:** Both training and validation loss decreased steadily, indicating **no overfitting**.  
* **Deployment Ready:** The model can now be saved and used to predict the authenticity of new, unseen Nepali news headlines or articles.  

---

### **Summary Table**

| Step | Action              | Tool Used          |
|------|---------------------|--------------------|
| 1    | Install Dependencies| `pip`, `transformers` |
| 2    | Load Model/Tokenizer| `xlm-roberta-base` |
| 3    | Preprocess Data     | `tokenizer.map()`  |
| 4    | Fine-tune           | `Trainer API`      |
| 5    | Evaluate            | `Accuracy`, `F1 Score` |
