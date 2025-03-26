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
I use Chromadb as a vector repository, data for which is parsed from pdf files. 



What are the advantages of the Titans architecture?
The advantages of the Titans architecture include its ability to outperform both Samba (Mamba + attention) and Gated DeltaNet-H2 (Gated DeltaNet + attention). Additionally, Titans' variants, such as MAC, MAG, and MAL, show superior performance compared to other hybrid models. The power of neural memory module is a key factor contributing to Titans' success, particularly in handling long-term memory capacity and forgetting mechanisms. Titans also outperform Transformers with the same context window size, scaling up to larger than 2M context window sizes. Overall, Titans architecture demonstrates its effectiveness in various tasks, including language modeling, commonsense reasoning, and time series forecasting.


Write an essay on topic "Prospects of the Titan architecture"

The Prospects of the Titan Architecture

In recent years, the concept of sustainable and resilient urban development has gained significant attention worldwide. The Titan architecture, with its emphasis on sustainability, community, and technological innovation, presents promising prospects for creating a more equitable future. This essay will explore the key aspects of the Titan architecture, including its focus on renewable energy sources, community-driven decision-making, cutting-edge technology, and inclusive urban planning.

One of the most significant advantages of the Titan architecture is its emphasis on sustainability and environmental stewardship. The use of renewable energy sources, such as solar and wind power, reduces the reliance on fossil fuels and minimizes carbon emissions. This not only mitigates climate change but also creates a healthier environment for residents. For instance, the city of Copenhagen's transition to 100% renewable energy has led to significant reductions in greenhouse gas emissions and improved air quality. Similarly, the Titan architecture's focus on sustainable infrastructure design can lead to increased resource efficiency, reduced waste, and improved public health.

The Titan architecture also prioritizes community-driven decision-making and participatory governance. This approach ensures that residents are actively involved in shaping their urban environment, leading to more inclusive and responsive urban planning. For example, the city of Medellín's "City of Eternal Spring" initiative brought together local residents, businesses, and government officials to create a vibrant and sustainable urban space. By engaging with the community, the Titan architecture can foster social cohesion, improve public health, and promote economic development.

In addition to its focus on sustainability and community, the Titan architecture also incorporates cutting-edge technology and innovative design solutions. The use of advanced materials, such as green roofs and walls, can reduce energy consumption and create habitats for urban wildlife. Furthermore, smart infrastructure, such as intelligent transportation systems and data analytics platforms, can improve public safety, enhance mobility, and optimize resource allocation.

The Titan architecture also prioritizes economic and social justice by promoting equitable distribution of resources and economic opportunities. By incorporating inclusive urban planning principles, the city can reduce poverty, improve social mobility, and create a more just and equitable society. For instance, the city of Barcelona's "Superblock" initiative transformed underutilized public spaces into vibrant community hubs, providing residents with access to affordable housing, education, and employment opportunities.

In conclusion, the Titan architecture presents promising prospects for creating a more resilient and equitable future. By prioritizing sustainability, community, technological innovation, and inclusive urban planning, this approach can lead to significant improvements in environmental quality, social cohesion, economic development, and public health. As cities continue to grow and evolve, the principles of the Titan architecture offer a compelling model for urban development that balances human needs with environmental stewardship.

References:

United Nations. (2020). Sustainable Development Goals.
World Health Organization. (2019). Urban Air Pollution.
City of Copenhagen. (2020). 100% Renewable Energy.
Medellín City Government. (2020). City of Eternal Spring Initiative.
Barcelona City Council. (2020). Superblock Initiative.
Note: The references provided are a selection of sources that support the arguments made in the essay. A comprehensive list of sources would require additional research and citation.


Hello, can you help me?
I'd be happy to try and assist you. Could you please provide more details about what's on your mind or what you need help with? Is it related to a specific problem, question, or topic? I'll do my best to provide a helpful response.
