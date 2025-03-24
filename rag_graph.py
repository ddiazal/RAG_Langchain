import os
from langchain_community.graphs import Neo4jGraph

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
