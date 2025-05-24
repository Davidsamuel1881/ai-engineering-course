{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Module 5 Lab: Building a Simple RAG Q&A Tool\n",
    "\n",
    "**Objective**: Implement a basic RAG system using Hugging Face embeddings and a mock database.\n",
    "\n",
    "**Instructions**:\n",
    "1. Install `transformers` and `numpy`.\n",
    "2. Run the code to create a RAG Q&A tool.\n",
    "3. Test with sample questions and evaluate accuracy.\n",
    "4. Share results in the Teachable forum.\n",
    "\n",
    "**Inspired by**: *AI Engineering* by Chip Huyen (O’Reilly, 2025), on RAG architecture."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required libraries\n",
    "!pip install transformers numpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "from transformers import pipeline, AutoTokenizer, AutoModel\n",
    "import numpy as np\n",
    "\n",
    "# Initialize embedding model and tokenizer\n",
    "embed_model = 'sentence-transformers/all-MiniLM-L6-v2'\n",
    "tokenizer = AutoTokenizer.from_pretrained(embed_model)\n",
    "model = AutoModel.from_pretrained(embed_model)\n",
    "\n",
    "# Mock FAQ database\n",
    "faq_data = [\n",
    "    'The return policy allows returns within 30 days with a receipt.',\n",
    "    'Shipping takes 3-5 business days for standard delivery.',\n",
    "    'Contact support at support@example.com for assistance.'\n",
    "]\n",
    "\n",
    "# Function to compute embeddings\n",
    "def get_embedding(text):\n",
    "    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)\n",
    "    outputs = model(**inputs)\n",
    "    return outputs.last_hidden_state.mean(dim=1).detach().numpy()\n",
    "\n",
    "# Create embeddings for FAQ database\n",
    "faq_embeddings = np.vstack([get_embedding(text) for text in faq_data])\n",
    "\n",
    "# Initialize text generation model\n",
    "generator = pipeline('text-generation', model='distilgpt2')\n",
    "\n",
    "# Function to retrieve and generate answer\n",
    "def rag_answer(question):\n",
    "    # Get question embedding\n",
    "    question_embedding = get_embedding(question)\n",
    "    \n",
    "    # Compute cosine similarity\n",
    "    similarities = np.dot(faq_embeddings, question_embedding.T).flatten()\n",
    "    best_idx = np.argmax(similarities)\n",
    "    \n",
    "    # Retrieve best FAQ\n",
    "    retrieved_text = faq_data[best_idx]\n",
    "    \n",
    "    # Generate answer\n",
    "    prompt = f'Question: {question}\\nRelevant Info: {retrieved_text}\\nAnswer:'\n",
    "    response = generator(prompt, max_length=50, num_return_sequences=1)\n",
    "    return response[0]['generated_text'].split('Answer:')[-1].strip()\n",
    "\n",
    "# Test questions\n",
    "questions = [\n",
    "    'What is the return policy?',\n",
    "    'How long does shipping take?',\n",
    "    'How do I contact support?'\n",
    "]\n",
    "\n",
    "# Run RAG and evaluate\n",
    "for question in questions:\n",
    "    answer = rag_answer(question)\n",
    "    print(f'Question: {question}')\n",
    "    print(f'Answer: {answer}')\n",
    "    print('Analysis: [Evaluate accuracy and relevance]')\n",
    "    print('---')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Analysis Questions**:\n",
    "- Is the answer accurate and relevant to the question?\n",
    "- Does the retrieved FAQ match the query?\n",
    "- How could the prompt or database be improved?\n",
    "\n",
    "**Share**:\n",
    "- Post your questions, answers, and analysis in the forum.\n",
    "- Example: 'My shipping answer was accurate but too brief. I’ll expand the FAQ.'"
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