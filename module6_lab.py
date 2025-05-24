{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Module 6 Lab: Fine-Tuning a Model for Sentiment Analysis\n",
    "\n",
    "**Objective**: Fine-tune DistilBERT for sentiment analysis using Hugging Face Transformers.\n",
    "\n",
    "**Instructions**:\n",
    "1. Install `transformers`, `datasets`, and `torch`.\n",
    "2. Run the code to fine-tune and evaluate the model.\n",
    "3. Share accuracy and insights in the Teachable forum.\n",
    "\n",
    "**Inspired by**: *AI Engineering* by Chip Huyen (O’Reilly, 2025), on fine-tuning."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required libraries\n",
    "!pip install transformers datasets torch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments\n",
    "from datasets import load_dataset\n",
    "import torch\n",
    "\n",
    "# Load dataset (IMDb subset)\n",
    "dataset = load_dataset('imdb', split={'train': 'train[:1000]', 'test': 'test[:200]'})  # Small subset for demo\n",
    "\n",
    "# Initialize tokenizer and model\n",
    "model_name = 'distilbert-base-uncased'\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)\n",
    "\n",
    "# Tokenize dataset\n",
    "def tokenize_function(examples):\n",
    "    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)\n",
    "\n",
    "tokenized_datasets = dataset.map(tokenize_function, batched=True)\n",
    "tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')\n",
    "tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])\n",
    "\n",
    "# Define training arguments\n",
    "training_args = TrainingArguments(\n",
    "    output_dir='./results',\n",
    "    num_train_epochs=1,  # Short for demo\n",
    "    per_device_train_batch_size=8,\n",
    "    per_device_eval_batch_size=8,\n",
    "    evaluation_strategy='epoch',\n",
    "    save_strategy='epoch',\n",
    "    logging_dir='./logs',\n",
    ")\n",
    "\n",
    "# Initialize trainer\n",
    "trainer = Trainer(\n",
    "    model=model,\n",
    "    args=training_args,\n",
    "    train_dataset=tokenized_datasets['train'],\n",
    "    eval_dataset=tokenized_datasets['test'],\n",
    ")\n",
    "\n",
    "# Fine-tune model\n",
    "trainer.train()\n",
    "\n",
    "# Evaluate model\n",
    "eval_results = trainer.evaluate()\n",
    "print(f'Evaluation Results: {eval_results}')\n",
    "\n",
    "# Test with sample text\n",
    "sample_text = 'This movie was fantastic and uplifting!'\n",
    "inputs = tokenizer(sample_text, return_tensors='pt', truncation=True, padding=True)\n",
    "outputs = model(**inputs)\n",
    "prediction = torch.argmax(outputs.logits, dim=1).item()\n",
    "print(f'Sample Text: {sample_text}')\n",
    "print(f'Prediction: {\"Positive\" if prediction == 1 else \"Negative\"}')\n",
    "print('Analysis: [Evaluate model accuracy and performance]')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Analysis Questions**:\n",
    "- What is the model’s evaluation accuracy?\n",
    "- Is the sample prediction correct?\n",
    "- How could the dataset or training be improved?\n",
    "\n",
    "**Share**:\n",
    "- Post your accuracy, sample prediction, and analysis in the forum.\n",
    "- Example: 'My model got 85% accuracy but struggled with short reviews. I’ll increase epochs.'"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}