import os
from typing import Any

import openai
from langchain_openai import ChatOpenAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_experimental.graph_transformer import LLMGraphTransformer
# Import unstructured text loader
from langchain_community.document_loaders import UnstructuredHTMLLoader
# Import text splitter
from langchain_text_splitters import TokenTextSplitter
# Import graph database instance
from langchain_community.graphs import Neo4jGraph

# Get the OpenAI secret key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create a document loader for unstructured HTML
html_loader = UnstructuredHTMLLoader("<html_file>")
# Load data from a HTML file
document = html_loader.load()

# Split data into chunks
splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)

# Instantiate LLM
llm = ChatOpenAI(
    api_key=openai_api_key,
    temperature=0.05,
    model_name="gpt-4o-mini"
)

# Instantiate graph transformer passing LLM
llm_transformer = LLMGraphTransformer(llm=llm)
# Convert document to graph document
graph_documents = llm_transformer.convert_to_graph_documents(document)


################# Creating Graph Document to GraphDB ###############
# Get graph url
neo4j_url: str = os.environ.get("NEO4J_URL")
# Get graph username
neo4j_un: str = os.environ.get("NEO4J_USERNAME")
# Get graph password
neo4j_password: str = os.environ.get("NEO4J_PASSWORD")

# Instantiate the Neo4j graph
graph: Neo4jGraph = Neo4jGraph(
    url=neo4j_url,
    username=neo4j_un,
    password=neo4j_password
)

# Add the graph documents, sources, and include entity labels
graph.add_graph_documents(
    graph_documents,
    include_source=True,
    baseEntityLabel=True
)

# refresh graph schema
graph.refresh_schema()

# Print the graph schema
print(graph.get_schema)

# Query the graph
#results = graph.query("""
#MATCH (relativity:Concept {id: "Theory Of Relativity"}) <-[:KNOWN_FOR]- (scientist)
#return scientist
#""")

###################### Querying the graph #########################
# Create the Graph Cypher QA chain
graph_qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    # Filtering the graph to improve model accuracy
    # exclude_types=["type"],
    # Validating the Cypher query
    # validate_cypher=True,
    # Adding few-shot examples
    # cypher_prompt="cypher_prompt_template",
    verbose=True,
)
# Invoke the chain with the input provided
results = graph_qa_chain.invoke(
    {
        "query": "What is the more accurate model?"
    }
)

# Print the result text
print(f"Final answer: {results['result']}")
