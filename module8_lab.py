{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Module 8 Lab: Evaluating a Sentiment Analysis Model\n",
    "\n",
    "**Objective**: Evaluate a DistilBERT model for sentiment analysis using quantitative and qualitative methods.\n",
    "\n",
    "**Instructions**:\n",
    "1. Install `transformers`, `datasets`, and `scikit-learn`.\n",
    "2. Run the code to compute metrics and analyze outputs.\n",
    "3. Share results in the Teachable forum.\n",
    "\n",
    "**Inspired by**: *AI Engineering* by Chip Huyen (O’Reilly, 2025), on evaluation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required libraries\n",
    "!pip install transformers datasets scikit-learn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline\n",
    "from datasets import load_dataset\n",
    "from sklearn.metrics import accuracy_score, precision_recall_fscore_support\n",
    "import numpy as np\n",
    "\n",
    "# Load dataset (IMDb subset)\n",
    "dataset = load_dataset('imdb', split='test[:200]')  # Small subset for demo\n",
    "\n",
    "# Initialize model and tokenizer\n",
    "model_name = 'distilbert-base-uncased-finetuned-sst-2-english'\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "model = AutoModelForSequenceClassification.from_pretrained(model_name)\n",
    "sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)\n",
    "\n",
    "# Predict sentiments\n",
    "texts = dataset['text']\n",
    "true_labels = dataset['label']  # 0: negative, 1: positive\n",
    "predictions = sentiment_pipeline(texts)\n",
    "pred_labels = [1 if pred['label'] == 'POSITIVE' else 0 for pred in predictions]\n",
    "\n",
    "# Compute quantitative metrics\n",
    "accuracy = accuracy_score(true_labels, pred_labels)\n",
    "precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')\n",
    "\n",
    "print('Quantitative Metrics:')\n",
    "print(f'Accuracy: {accuracy:.4f}')\n",
    "print(f'Precision: {precision:.4f}')\n",
    "print(f'Recall: {recall:.4f}')\n",
    "print(f'F1 Score: {f1:.4f}')\n",
    "\n",
    "# Qualitative analysis on sample texts\n",
    "sample_texts = [\n",
    "    'This movie was absolutely fantastic!',\n",
    "    'The plot was confusing and boring.',\n",
    "    'Great acting, but the story felt biased against certain groups.'\n",
    "]\n",
    "\n",
    "print('\\nQualitative Analysis:')\n",
    "for text in sample_texts:\n",
    "    result = sentiment_pipeline(text)[0]\n",
    "    print(f'Text: {text}')\n",
    "    print(f'Prediction: {result[\"label\"]} (Score: {result[\"score\"]:.4f})')\n",
    "    print('Analysis: [Evaluate coherence, fairness, reliability]')\n",
    "    print('---')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Analysis Questions**:\n",
    "- Are the quantitative metrics (accuracy, F1) satisfactory?\n",
    "- Do sample predictions align with expectations?\n",
    "- Is there evidence of bias or unreliability in the outputs?\n",
    "\n",
    "**Share**:\n",
    "- Post your metrics, sample predictions, and analysis in the forum.\n",
    "- Example: 'My model had 88% accuracy but misclassified biased text. I’ll test more diverse inputs.'"
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