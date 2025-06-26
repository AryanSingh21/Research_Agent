# AI Research Assistant Agent
# A sophisticated multi-tool agent using LangGraph, Groq, and DuckDuckGo for research tasks

import os
import re
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# LangChain and LangGraph imports for agent-based workflows
from langchain.chat_models import init_chat_model
from langchain.tools import DuckDuckGoSearchRun, Tool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Document processing for handling web content
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Arxiv API for searching and retrieving academic papers
import arxiv

@dataclass
class PaperInfo:
    """Structure to hold information about academic papers"""
    title: str
    authors: List[str]
    published: datetime
    summary: str
    pdf_url: str
    citation_count: Optional[int] = None
    arxiv_id: Optional[str] = None

class ArxivSearchTool:
    """Custom tool for searching and retrieving papers from ArXiv"""
    
    def __init__(self):
        # Initialize ArXiv client for API interactions
        self.client = arxiv.Client()
        self.name = "arxiv_search"
        self.description = "Search for recent research papers on ArXiv. Input should be a research topic or keywords."
    
    def __call__(self, query: str) -> str:
        """Search for papers on ArXiv and return formatted results"""
        try:
            # Configure search parameters for ArXiv
            search = arxiv.Search(
                query=query,
                max_results=10,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            # Iterate through search results to collect paper information
            for result in self.client.results(search):
                paper = PaperInfo(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    published=result.published,
                    summary=result.summary,
                    pdf_url=result.pdf_url,
                    arxiv_id=result.entry_id.split('/')[-1]
                )
                papers.append(paper)
            
            # Handle case where no papers are found
            if not papers:
                return "No papers found for the given query."
            
            # Format results for display
            result_text = f"Found {len(papers)} recent papers on '{query}':\n\n"
            for i, paper in enumerate(papers[:5], 1):
                result_text += f"{i}. **{paper.title}**\n"
                result_text += f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}\n"
                result_text += f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
                result_text += f"   ArXiv ID: {paper.arxiv_id}\n"
                result_text += f"   Abstract: {paper.summary[:200]}...\n\n"
            
            # Store papers for use by other tools
            self._last_search_papers = papers
            return result_text
            
        except Exception as e:
            return f"Error searching ArXiv: {str(e)}"
    
    def get_last_papers(self) -> List[PaperInfo]:
        """Retrieve papers from the last search"""
        return getattr(self, '_last_search_papers', [])

class PaperSummarizerTool:
    """Tool for generating detailed summaries of research papers"""
    
    def __init__(self, model, arxiv_tool):
        # Initialize with the language model and ArXiv tool
        self.model = model
        self.arxiv_tool = arxiv_tool
        self.name = "paper_summarizer"
        self.description = "Generate detailed summaries of the most recent papers found. Use after searching for papers."
    
    def __call__(self, query: str) -> str:
        """Generate comprehensive summaries of recent papers"""
        # Retrieve papers from the last ArXiv search
        papers = self.arxiv_tool.get_last_papers()
        if not papers:
            return "No papers available to summarize. Please search for papers first using arxiv_search."
        
        # Summarize the top 3 papers
        top_papers = papers[:3]
        summaries = []
        
        for i, paper in enumerate(top_papers, 1):
            # Create a prompt for summarizing each paper
            summary_prompt = f"""
            Generate a comprehensive but concise summary of this research paper:
            
            Title: {paper.title}
            Authors: {', '.join(paper.authors)}
            Published: {paper.published.strftime('%Y-%m-%d')}
            
            Abstract: {paper.summary}
            
            Provide a structured summary including:
            1. Main research question/problem addressed
            2. Key methodology/approach used
            3. Primary findings and contributions
            4. Significance and implications for the field
            
            Keep the summary under 200 words but comprehensive.
            """
            
            try:
                # Invoke the model to generate the summary
                response = self.model.invoke([HumanMessage(content=summary_prompt)])
                summary_text = response.content if hasattr(response, 'content') else str(response)
                summaries.append(f"**Paper {i}: {paper.title}**\n\n{summary_text}\n\n---\n")
            except Exception as e:
                summaries.append(f"**Paper {i}: {paper.title}**\n\nError generating summary: {str(e)}\n\n---\n")
        
        return "\n".join(summaries)

class TrendAnalyzerTool:
    """Tool for analyzing publication trends and creating visualizations"""
    
    def __init__(self, arxiv_tool):
        # Initialize with the ArXiv tool for accessing paper data
        self.arxiv_tool = arxiv_tool
        self.name = "trend_analyzer"
        self.description = "Analyze publication trends and create visualizations for a research topic. Use after searching for papers."
    
    def __call__(self, query: str) -> str:
        """Analyze publication trends and create visualization"""
        try:
            # Configure search parameters for trend analysis
            search = arxiv.Search(
                query=query,
                max_results=50,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            client = arxiv.Client()
            # Collect paper data for trend analysis
            for result in client.results(search):
                paper = PaperInfo(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    published=result.published,
                    summary=result.summary,
                    pdf_url=result.pdf_url,
                    arxiv_id=result.entry_id.split('/')[-1]
                )
                papers.append(paper)
            
            # Handle case where no papers are found
            if not papers:
                return "No papers found for trend analysis."
            
            # Extract publication dates for trend analysis
            dates = [paper.published for paper in papers]
            df = pd.DataFrame({'date': dates})
            df['year_month'] = df['date'].dt.to_period('M')
            
            # Count papers per month for trend visualization
            trend_data = df.groupby('year_month').size().reset_index(name='count')
            trend_data['date'] = trend_data['year_month'].dt.to_timestamp()
            
            # Create a line plot for publication trends
            plt.figure(figsize=(12, 6))
            plt.plot(trend_data['date'], trend_data['count'], marker='o', linewidth=2, markersize=6)
            plt.title(f'Publication Trend: {query}', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Number of Papers', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the trend visualization as a PNG file
            filename = f"publication_trend_{query.replace(' ', '_').lower()}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Analyze author contributions and collaboration
            all_authors = []
            for paper in papers:
                all_authors.extend(paper.authors)
            
            author_counts = pd.Series(all_authors).value_counts()
            collaboration_rate = sum(len(paper.authors) > 1 for paper in papers) / len(papers)
            
            # Compile the trend analysis results
            result = f"📈 **Publication Trend Analysis for '{query}'**\n\n"
            result += f"✅ Visualization saved as: {filename}\n\n"
            result += f"**Analysis Summary:**\n"
            result += f"- Total papers analyzed: {len(papers)}\n"
            result += f"- Unique authors: {len(author_counts)}\n"
            result += f"- Collaboration rate: {collaboration_rate:.2%}\n"
            result += f"- Most prolific authors: {', '.join(author_counts.head(3).index.tolist())}\n"
            result += f"- Publication timespan: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}\n"
            
            return result
            
        except Exception as e:
            return f"Error analyzing publication trends: {str(e)}"

class DatasetRecommenderTool:
    """Tool for recommending relevant datasets for research topics"""
    
    def __init__(self, model, arxiv_tool):
        # Initialize with the language model and ArXiv tool
        self.model = model
        self.arxiv_tool = arxiv_tool
        self.name = "dataset_recommender"
        self.description = "Recommend relevant open-source datasets for a research topic based on recent papers."
    
    def __call__(self, query: str) -> str:
        """Recommend a relevant dataset for the research topic"""
        # Retrieve papers from the last ArXiv search for context
        papers = self.arxiv_tool.get_last_papers()
        
        if papers:
            paper_context = "\n".join([f"- {paper.title}: {paper.summary[:150]}..." for paper in papers[:3]])
        else:
            paper_context = "No specific papers available for context."
        
        # Create a prompt for dataset recommendation
        recommendation_prompt = f"""
        Based on the research topic "{query}" and these recent papers:
        
        {paper_context}
        
        Recommend ONE specific, high-quality open-source dataset that would be valuable for researchers in this area.
        
        Provide a comprehensive recommendation including:
        1. **Dataset Name and Source**
        2. **Description** (what it contains, scope, purpose)
        3. **Relevance** (why it's perfect for this research area)
        4. **Access Information** (URL, installation instructions)
        5. **Key Statistics** (size, format, number of samples, etc.)
        6. **Usage Notes** (any important considerations)
        
        Focus on well-established, actively maintained datasets that are widely used in the research community.
        Be specific and actionabl in your recommendation.
        """
        
        try:
            # Invoke the model to generate the dataset recommendation
            response = self.model.invoke([HumanMessage(content=recommendation_prompt)])
            recommendation = response.content if hasattr(response, 'content') else str(response)
            
            result = f"🗃️ **Dataset Recommendation for '{query}'**\n\n"
            result += recommendation
            
            return result
            
        except Exception as e:
            return f"Error generating dataset recommendation: {str(e)}"

class ResearchAssistantAgent:
    """Main AI Research Assistant Agent using LangGraph for orchestration"""
    
    def __init__(self, groq_api_key: str):
        # Initialize the Groq model using LangChain's chat model interface
        self.model = init_chat_model(
            model="deepseek-r1-distill-llama-70b",
            model_provider="groq",
            api_key=groq_api_key,
            temperature=0.1
        )
        
        # Initialize memory to store conversation context
        self.memory = MemorySaver()
        
        # Initialize custom tools for research tasks
        self.arxiv_tool = ArxivSearchTool()
        self.summarizer_tool = PaperSummarizerTool(self.model, self.arxiv_tool)
        self.trend_tool = TrendAnalyzerTool(self.arxiv_tool)
        self.dataset_tool = DatasetRecommenderTool(self.model, self.arxiv_tool)
        
        # Initialize DuckDuckGo for web searches
        self.web_search = DuckDuckGoSearchRun()
        
        # Create LangChain Tool objects for use in the agent
        self.tools = self._create_langchain_tools()
        
        # Create the ReAct agent with LangGraph
        self.agent_executor = create_react_agent(
            model=self.model, 
            tools=self.tools, 
            checkpointer=self.memory
        )
        
        # Configuration for maintaining conversation sessions
        self.config = {"configurable": {"thread_id": "research_session"}}
    
    def _create_langchain_tools(self) -> List[Tool]:
        """Create LangChain Tool objects from custom tools and web search"""
        
        def web_search_wrapper(query: str) -> str:
            """Wrapper function for performing web searches using DuckDuckGo"""
            try:
                result = self.web_search.run(query)
                return f"🌐 **Web Search Results for '{query}':**\n\n{result}"
            except Exception as e:
                return f"Error in web search: {str(e)}"
        
        # Return a list of LangChain Tool objects
        return [
            Tool(
                name=self.arxiv_tool.name,
                description=self.arxiv_tool.description,
                func=self.arxiv_tool
            ),
            Tool(
                name=self.summarizer_tool.name,
                description=self.summarizer_tool.description,
                func=self.summarizer_tool
            ),
            Tool(
                name=self.trend_tool.name,
                description=self.trend_tool.description,
                func=self.trend_tool
            ),
            Tool(
                name=self.dataset_tool.name,
                description=self.dataset_tool.description,
                func=self.dataset_tool
            ),
            Tool(
                name="web_search",
                description="Search the web for additional information, news, or resources. Use for supplementary information beyond academic papers.",
                func=web_search_wrapper
            )
        ]
    
    def process_query(self, query: str):
        """Process a research query using the ReAct agent and yield responses"""
        # Create the input message for the agent
        input_message = HumanMessage(content=query)
        
        try:
            # Stream the agent's responses for real-time output
            for step in self.agent_executor.stream({"messages": [input_message]}, config=self.config, stream_mode="values"):
                step["messages"][-1].pretty_print()
                yield step["messages"][-1].content
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def chat(self, message: str) -> str:
        """Handle conversational interactions with the agent"""
        return self.process_query(message)
    
    def reset_conversation(self):
        """Reset the conversation memory by creating a new session ID"""
        self.config = {"configurable": {"thread_id": f"research_session_{datetime.now().timestamp()}"}}

def main():
    """Main function to demonstrate the agent's functionality"""
    
    # Initialize the agent with the Groq API key
    groq_api_key = "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" # This key will be revoked in 4 days after evaluation
    if not groq_api_key:
        print("Please set your GROQ_API_KEY")
        return
    
    agent = ResearchAssistantAgent(groq_api_key)
    
    # Define example queries for demonstration
    example_queries = [
        "Summarize the three latest papers on LLM safety",
        "Recommend one open-source dataset",
        "Plot the publication trend in this topic over the past 5 years."
    ]
    
    # Display welcome message
    print("🤖 AI Research Assistant Agent Ready!")
    print("Built with LangGraph's ReAct Agent Framework")
    print("="*60)
    
    while True:
        # Display menu options for user interaction
        print("\nOptions:")
        print("1-3: Run example queries")
        print("chat: Start conversation mode")
        print("reset: Reset conversation")
        print("q: Quit")
        
        choice = input("\nYour choice: ").strip().lower()
        
        # Handle user input
        if choice == 'q':
            break
        elif choice == 'reset':
            agent.reset_conversation()
            print("✅ Conversation reset")
            continue
        elif choice == 'chat':
            print("\n💬 Conversation mode (type 'exit' to return to menu)")
            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == 'exit':
                    break
                if user_input:
                    # Process user input in conversation mode
                    response = agent.chat(user_input)
                    # for con in response:
                        # print(f"\n🤖 Agent: {con}")
            continue
        
        # Handle example queries or custom input
        query = None
        if choice in ['1', '2', '3']:
            idx = int(choice) - 1
            if 0 <= idx < len(example_queries):
                query = example_queries[idx]
        else:
            query = choice
        
        if query:
            print(f"\n🔍 Processing: {query}")
            print("="*60)
            
            # Process the query and display results
            result = agent.chat(query)
            # print(f"\n📊 Result:\n{result}")
            for con in result:
                print(f"\n🤖 Agent: {con}")
            print("="*60)

if __name__ == "__main__":
    main()