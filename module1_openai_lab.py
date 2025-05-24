{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Module 1 Lab: Text Generation with OpenAI API\n",
    "\n",
    "**Objective**: Use the OpenAI API to generate text with GPT-3.5, exploring foundation models.\n",
    "\n",
    "**Instructions**:\n",
    "1. Install the `openai` library.\n",
    "2. Replace `YOUR_API_KEY` with your OpenAI API key.\n",
    "3. Run the code to generate text.\n",
    "4. Share your output in the Teachable forum.\n",
    "\n",
    "**Inspired by**: *AI Engineering* by Chip Huyen (O’Reilly, 2025), focusing on accessible APIs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install openai library\n",
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
    "# Define prompt\n",
    "prompt = 'The future of AI engineering is'\n",
    "\n",
    "# Call OpenAI API\n",
    "response = openai.Completion.create(\n",
    "    model='text-davinci-003',  # GPT-3.5 model\n",
    "    prompt=prompt,\n",
    "    max_tokens=50,\n",
    "    temperature=0.7\n",
    ")\n",
    "\n",
    "# Print generated text\n",
    "print('Generated Text:', response.choices[0].text.strip())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Explore Further**:\n",
    "- Change the prompt to something like 'Write a tagline for an AI startup.'\n",
    "- Adjust `temperature` (0.0 to 1.0) to control creativity.\n",
    "- Share your results in the forum!"
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