# MultiAgenticRAG-for-research
This project features my implementation of a Deep Researcher-like system - a Multi-agent Retrieval-Augmented Generation (RAG) application that capable of:
  + answer to general user query (chatting)
  + answering user queries with document retrieval from PDF files or web search (RAG+webRAG)
  + generating concise essays on user-specified topics
    
This application built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Ollama](https://github.com/ollama/ollama). Before launching, you need to install Ollama and pull two models: [Llama3.2](https://ollama.com/library/llama3.2) as llm and [nomic-embed-text](https://ollama.com/library/nomic-embed-text) as embedding model for RAG.
```
ollama pull ollama run llama3.2:3b
ollama pull nomic-embed-text
```
## Getting Started
To get started with this project, follow these steps:

1) clone the repository to your local machine:
```
git clone https://github.com/Dortp68/AgenticRAG-for-research
```
2) Go to the directory and install the dependencies
```
pip install -r requirements.txt
```
3) Run application
```
python3 -m app
```
## Graph Architecture
![graph](https://github.com/Dortp68/AgenticRAG-for-research/blob/main/imgs/Graph.drawio%20(1).png)

The graph architecture is a main graph that can call tools and subgraphs wrapped in the ```@tool``` decorator, they can be called as tools. Since in ollama the invocation of tools is implemented in such a way that at least one tool must be invoked, I created a separate tool responsible for chatting with the user. Also ```essay writer``` calls the ```research assistant``` inside to search for information in vectorstore or on the internet.

```research assistant``` using websearch with Duckduckgo search, parse html content and inserts it into the prompt context for llm. Within the graph, a relevance check of the retrieved documents and a hallucination check are performed.

```essay writer``` makes an essay outline, generates additional queries, sends it to the ```research assistant``` and composes an essay based on the information received.

```chat tool``` responds to common user requests. Shares memory (persist memory) with the main graph and performs simple memory management.

## Retriever
I use Chromadb as a vector repository, data for which is parsed from pdf files. For RAG, an ensemble of retrievers with reranking is used:
+ BM25Retriever
+ Similarity search retriever
+ mmr retriever
+ cross-encoder/ms-marco-MiniLM-L-6-v2 for reranking
## Functionality
![](https://github.com/Dortp68/AgenticRAG-for-research/blob/main/imgs/Screenshot%20from%202025-03-26%2008-25-08.png)![](https://github.com/Dortp68/AgenticRAG-for-research/blob/main/imgs/Screenshot%20from%202025-03-26%2008-27-39.png)

I used gradio to build the UI of application. You can:
+ download PDFs
+ change top_k param
+ enable/disable reranking
+ enable/disable hallucination check
+ TODO: voice input
