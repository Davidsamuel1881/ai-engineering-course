{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Module 4 Lab: Building a Text Summarization Tool\n",
    "\n",
    "**Objective**: Use OpenAI API to create a text summarization tool with prompt engineering.\n",
    "\n",
    "**Instructions**:\n",
    "1. Install `openai`.\n",
    "2. Replace `YOUR_API_KEY` with your OpenAI API key.\n",
    "3. Run the code to summarize the sample text.\n",
    "4. Evaluate the summary and share in the Teachable forum.\n",
    "\n",
    "**Inspired by**: *AI Engineering* by Chip Huyen (O’Reilly, 2025), on prompt engineering."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install openai\n",
    "!pip install openai"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import openai\n",
    "import openai\n",
    "\n",
    "# Set up API key\n",
    "openai.api_key = 'YOUR_API_KEY'  # Replace with your OpenAI API key\n",
    "\n",
    "# Sample text to summarize\n",
    "sample_text = '''\n",
    "Artificial intelligence is transforming industries worldwide. In healthcare, AI helps diagnose diseases by analyzing medical images with high accuracy. In finance, it detects fraudulent transactions by identifying unusual patterns. However, challenges like data privacy and ethical concerns must be addressed to ensure responsible use.\n",
    "'''\n",
    "\n",
    "# Define prompt\n",
    "prompt = f'Summarize the following text in 30 words or less, focusing on key points:\\n\\n{sample_text}'\n",
    "\n",
    "# Call OpenAI API\n",
    "response = openai.Completion.create(\n",
    "    model='text-davinci-003',\n",
    "    prompt=prompt,\n",
    "    max_tokens=50,\n",
    "    temperature=0.5\n",
    ")\n",
    "\n",
    "# Print summary\n",
    "summary = response.choices[0].text.strip()\n",
    "print('Original Text:', sample_text)\n",
    "print('Summary:', summary)\n",
    "print('Analysis: [Evaluate accuracy, brevity, and clarity]')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Analysis Questions**:\n",
    "- Does the summary capture the main points?\n",
    "- Is it concise and clear?\n",
    "- How could the prompt be improved?\n",
    "\n",
    "**Share**:\n",
    "- Post your summary and prompt in the forum.\n",
    "- Example: 'My summary was concise but missed finance details. I’ll add “include all industries” to the prompt.'"
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