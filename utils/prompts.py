RETRIEVER_TOOL_PROMPT = "Use this if you need to search information from research papers on neural network architectures, large language models, and new developments in this area. "

DOC_GRADER_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

RAG_PROMPT = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Adjust your response length based on the question, but use five sentences maximum and keep the answer concise.

Answer:"""

CHECK_HALLUCINATIONS = """You are a grader assessing whether an LLM generation is supported by a set of retrieved facts. 

Give a binary score between 'yes' or 'no', where 'yes' means that the answer is supported by the set of facts.

<Set of facts>
{documents}
<Set of facts/>


<LLM generation> 
{generation}
<LLM generation/> 


If the set of facts is not provided, give the score 'yes'.

"""

PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""

ROUTER_PROMPT = """Determine if the user query requires a simple, advanced or . An 'advanced' request might require \
multiple steps like retrieving an order ID and looking up shipping information, whereas a 'simple' request can \
handle more straightforward queries. return either 'simple' or 'advanced'. Do not explain your reasoning Your \
only task is to determine where to route the user query."""