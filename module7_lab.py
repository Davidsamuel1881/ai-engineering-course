{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Module 7 Lab: Building a Simple AI Agent with LangChain\n",
    "\n",
    "**Objective**: Create an AI agent using LangChain to answer questions with a mock tool.\n",
    "\n",
    "**Instructions**:\n",
    "1. Install `langchain`, `transformers`, and `openai`.\n",
    "2. Replace `YOUR_API_KEY` with your OpenAI API key.\n",
    "3. Run the code to test the agent.\n",
    "4. Share results in the Teachable forum.\n",
    "\n",
    "**Inspired by**: *AI Engineering* by Chip Huyen (O’Reilly, 2025), on AI agents."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required libraries\n",
    "!pip install langchain transformers openai"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "from langchain.agents import initialize_agent, Tool\n",
    "from langchain.llms import OpenAI\n",
    "import os\n",
    "\n",
    "# Set up OpenAI API key\n",
    "os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'  # Replace with your OpenAI API key\n",
    "\n",
    "# Mock tool for simulated web search\n",
    "def mock_search(query):\n",
    "    mock_data = {\n",
    "        'What is the capital of France?': 'The capital of France is Paris.',\n",
    "        'Who won the 2024 World Series?': 'The 2024 World Series was won by the Los Angeles Dodgers.',\n",
    "        'What is AI engineering?': 'AI engineering builds applications using foundation models.'\n",
    "    }\n",
    "    return mock_data.get(query, 'No information found.')\n",
    "\n",
    "# Define tools\n",
    "tools = [\n",
    "    Tool(\n",
    "        name='WebSearch',\n",
    "        func=mock_search,\n",
    "        description='Simulates a web search for factual questions.'\n",
    "    )\n",
    "]\n",
    "\n",
    "# Initialize LLM\n",
    "llm = OpenAI(temperature=0.7)\n",
    "\n",
    "# Initialize agent\n",
    "agent = initialize_agent(tools, llm, agent='zero-shot-react-description', verbose=True)\n",
    "\n",
    "# Test questions\n",
    "questions = [\n",
    "    'What is the capital of France?',\n",
    "    'Who won the 2024 World Series?',\n",
    "    'What is AI engineering?'\n",
    "]\n",
    "\n",
    "# Run agent and evaluate\n",
    "for question in questions:\n",
    "    response = agent.run(question)\n",
    "    print(f'Question: {question}')\n",
    "    print(f'Response: {response}')\n",
    "    print('Analysis: [Evaluate accuracy and tool usage]')\n",
    "    print('---')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Analysis Questions**:\n",
    "- Is the response accurate and relevant?\n",
    "- Did the agent use the mock search tool correctly?\n",
    "- How could the tool or prompt be improved?\n",
    "\n",
    "**Share**:\n",
    "- Post your questions, responses, and analysis in the forum.\n",
    "- Example: 'My agent answered correctly but was slow. I’ll optimize the tool.'"
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