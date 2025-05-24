{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Module 3 Lab: Analyzing Foundation Model Outputs\n",
    "\n",
    "**Objective**: Generate text with DistilBERT and analyze for limitations like biases or errors.\n",
    "\n",
    "**Instructions**:\n",
    "1. Install `transformers`.\n",
    "2. Run the code with three prompts.\n",
    "3. Analyze outputs for accuracy, coherence, or biases.\n",
    "4. Share findings in the Teachable forum.\n",
    "\n",
    "**Inspired by**: *AI Engineering* by Chip Huyen (O’Reilly, 2025), on model limitations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install transformers\n",
    "!pip install transformers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import pipeline\n",
    "from transformers import pipeline\n",
    "\n",
    "# Load DistilBERT model\n",
    "generator = pipeline('text-generation', model='distilbert-base-uncased')\n",
    "\n",
    "# Define prompts\n",
    "prompts = [\n",
    "    'The weather today is',\n",
    "    'AI engineering is exciting because',\n",
    "    'The best software engineers are'\n",
    "]\n",
    "\n",
    "# Generate and analyze outputs\n",
    "for prompt in prompts:\n",
    "    print(f'Prompt: {prompt}')\n",
    "    output = generator(prompt, max_length=30, num_return_sequences=1)\n",
    "    text = output[0]['generated_text']\n",
    "    print(f'Output: {text}')\n",
    "    print('Analysis: [Add your observations, e.g., coherence, bias]')\n",
    "    print('---')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Analysis Questions**:\n",
    "- Is the output coherent and relevant to the prompt?\n",
    "- Does the sensitive prompt ('best engineers') show biases (e.g., stereotypes)?\n",
    "- Are there any errors or hallucinations?\n",
    "\n",
    "**Share**:\n",
    "- Post your outputs and analysis in the forum.\n",
    "- Example: 'The engineer prompt mentioned 'male coders,' which seems biased.'"
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