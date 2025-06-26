# AI Research Assistant Agent

A sophisticated multi-tool AI agent built with LangChain, Groq, and multiple research tools to help founders and scientists with complex research queries.

## Features

- **ArXiv Paper Search**: Find and analyze recent academic papers
- **Intelligent Summarization**: Generate comprehensive paper summaries using Groq's Mixtral model
- **Web Search Integration**: DuckDuckGo search for additional context
- **Publication Trend Analysis**: Visualize research trends over time
- **Dataset Recommendations**: Suggest relevant open-source datasets
- **Multi-Tool Orchestration**: Intelligent tool selection and chaining

## Architecture

The agent uses **LangGraph's ReAct (Reasoning and Acting) framework** for superior reasoning capabilities:

1. **Query Analysis**: Uses advanced reasoning to break down complex requests
2. **Dynamic Tool Selection**: Intelligently chooses tools based on context and reasoning
3. **Iterative Execution**: Can use tools multiple times and reason about results
4. **Memory Integration**: Maintains conversation context across interactions
5. **Self-Correction**: Can adjust approach based on intermediate results

The system leverages:

- **LangGraph's create_react_agent** for robust agent orchestration
- **Groq's deepseek-r1-distill-llama-70b** for fast, high-quality reasoning and text processing
- **MemorySaver** for persistent conversation memory
- **ArXiv API** for academic paper retrieval
- **DuckDuckGo Search** for web information
- **Custom analysis tools** for data visualization and insights

## Setup Instructions

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd <folder>

# Create virtual environment
python -m venv research_agent_env
source research_agent_env/bin/activate  # On Windows: research_agent_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

```bash
GROQ_API_KEY=your_groq_api_key_here # I have already provided this on the code (Hard-coded) will be revoked automatically after 4 days
```

### 3. Run the Agent

```bash
python main.py
```

## Usage Examples

### Example 1: LLM Safety Research

```
"Summarize the three latest papers on LLM safety."
```

**Expected Output:**

- Summaries of 3 most recent LLM safety papers
- Detailed dataset recommendation with access instructions

### Example 2: AI Alignment Research

```
"Find recent papers on AI alignment, summarize the top 3."
```

**Expected Output:**

- ArXiv search results for AI alignment
- Comprehensive summaries of top papers
- Targeted dataset recommendation
- Source citations and access links

## Tool Capabilities

### 1. ArXiv Paper Search

- Searches recent academic papers by topic
- Retrieves metadata (authors, dates, abstracts)
- Sorts by publication date for latest research

### 2. Intelligent Summarization

- Generates structured summaries with:
  - Research question/problem
  - Methodology/approach
  - Key findings
  - Field significance

### 3. Publication Trend Analysis

- Creates time-series visualizations
- Analyzes author collaboration patterns
- Identifies research momentum

### 4. Dataset Recommendations

- Context-aware suggestions based on papers
- Includes access instructions and statistics
- Focuses on open-source, maintained datasets

### 5. Web Search Enhancement

- Supplements academic search with web context
- Finds additional resources and news
- Provides broader research landscape

## Sample Output

For the query: _"Summarize the three latest papers on LLM safety, recommend one open-source dataset, and plot the publication trend in this topic over the past year."_

```
📊 Found 10 recent papers on LLM safety:

1. **Constitutional AI: Harmlessness from AI Feedback**
   Authors: Anthropic Team
   Published: 2024-01-15
   ArXiv ID: 2401.09334

[... detailed summaries of top 3 papers ...]

📈 **Publication Trend Analysis:**
- Total papers analyzed: 47
- Unique authors: 156
- Collaboration rate: 89%
- Visualization saved as: publication_trend_llm_safety.png

🗃️ **Dataset Recommendation:**
**Anthropic's HH-RLHF Dataset**
- Source: Hugging Face (Anthropic/hh-rlhf)
- Contains: 161K human preference comparisons for helpfulness and harmlessness
- Relevance: Essential for training safer AI systems using RLHF
- Access: https://huggingface.co/datasets/Anthropic/hh-rlhf
- Size: ~500MB, JSON format
```

## Troubleshooting

### Common Issues

1. **Groq API Key Error**

   - Ensure API key is correctly set in environment
   - Check for typos in key
   - Verify account has API access
   - It is possible that API key may have access token error, due to multiple hits, please change the API_KEY in that scenarion

2. **ArXiv Search Timeout**

   - Reduce max_results parameter
   - Check internet connection
   - Try more specific search terms

3. **Visualization Not Saving**
   - Ensure write permissions in directory
   - Install matplotlib with GUI backend
   - Check for sufficient disk space

### Performance Tips

- Use specific search terms for better results
- Limit paper searches to 10-50 results for faster processing
- Run with verbose=False for cleaner output
- Use conda environment for better package management

## Contributing

This agent is designed to be extensible. To add new tools:

1. Create tool function following the pattern
2. Add to `_create_tools()` method
3. Update system prompt with new capabilities
4. Test with relevant queries

## License

Open source - feel free to modify and extend for your research needs.
