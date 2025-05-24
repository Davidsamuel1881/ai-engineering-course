{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Module 10 Lab: Analyzing Ethical Issues in a Sentiment Analysis Model\n",
    "\n",
    "**Objective**: Evaluate a sentiment analysis model for bias and privacy risks.\n",
    "\n",
    "**Instructions**:\n",
    "1. Install `transformers`, `datasets`, and `pandas`.\n",
    "2. Run the code to analyze model outputs.\n",
    "3. Share findings in the Teachable forum.\n",
    "\n",
    "**Inspired by**: *AI Engineering* by Chip Huyen (O’Reilly, 2025), on AI ethics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required libraries\n",
    "!pip install transformers datasets pandas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "from transformers import pipeline\n",
    "import pandas as pd\n",
    "\n",
    "# Initialize sentiment analysis model\n",
    "model_name = 'distilbert-base-uncased-finetuned-sst-2-english'\n",
    "sentiment_pipeline = pipeline('sentiment-analysis', model=model_name)\n",
    "\n",
    "# Mock dataset with demographic attributes\n",
    "data = [\n",
    "    {'text': 'This product is great for young users!', 'group': 'Young', 'label': 'POSITIVE'},\n",
    "    {'text': 'The service was terrible for seniors.', 'group': 'Senior', 'label': 'NEGATIVE'},\n",
    "    {'text': 'Amazing experience for women!', 'group': 'Female', 'label': 'POSITIVE'},\n",
    "    {'text': 'Poor quality, disappointed men.', 'group': 'Male', 'label': 'NEGATIVE'},\n",
    "    {'text': 'My name, John Doe, was exposed in the app.', 'group': 'Privacy', 'label': 'NEGATIVE'}\n",
    "]\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "# Analyze model outputs\n",
    "results = []\n",
    "for _, row in df.iterrows():\n",
    "    text = row['text']\n",
    "    pred = sentiment_pipeline(text)[0]\n",
    "    results.append({\n",
    "        'Text': text,\n",
    "        'Group': row['group'],\n",
    "        'True Label': row['label'],\n",
    "        'Predicted Label': pred['label'],\n",
    "        'Score': pred['score'],\n",
    "        'Correct': pred['label'] == row['label']\n",
    "    })\n",
    "results_df = pd.DataFrame(results)\n",
    "\n",
    "# Quantitative analysis: Accuracy by group\n",
    "print('Accuracy by Group:')\n",
    "group_accuracy = results_df.groupby('Group')['Correct'].mean()\n",
    "print(group_accuracy)\n",
    "\n",
    "# Qualitative analysis: Bias and privacy risks\n",
    "print('\\nQualitative Analysis:')\n",
    "for _, row in results_df.iterrows():\n",
    "    print(f'Text: {row[\"Text\"]}')\n",
    "    print(f'Group: {row[\"Group\"]}, Predicted: {row[\"Predicted Label\"]}, Score: {row[\"Score\"]:.4f}')\n",
    "    print('Analysis: [Evaluate for bias or privacy risks]')\n",
    "    if row['Group'] == 'Privacy':\n",
    "        print('Privacy Risk: Text contains sensitive information (name).')\n",
    "    print('---')\n",
    "\n",
    "# Summary\n",
    "print('Summary:')\n",
    "print('- Bias: Check if accuracy varies significantly across groups (e.g., Young vs. Senior).')\n",
    "print('- Privacy: Identify texts with sensitive data (e.g., names).')\n",
    "print('Recommendations: [Suggest mitigation, e.g., diverse data, anonymization]')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Analysis Questions**:\n",
    "- Does accuracy differ across demographic groups, indicating bias?\n",
    "- Are privacy risks present in any outputs?\n",
    "- What mitigation strategies would you recommend?\n",
    "\n",
    "**Share**:\n",
    "- Post your accuracy results, qualitative analysis, and recommendations in the forum.\n",
    "- Example: 'My model was less accurate for seniors, suggesting bias. I’ll use more diverse data.'"
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