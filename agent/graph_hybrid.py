import os
import json
from typing import TypedDict, List, Any, Optional
import dspy
from langgraph.graph import StateGraph, END

from agent.rag.retrieval import LocalRetriever
from agent.dspy_signatures import Router, Synthesizer, schema, optimization
from agent.tools.sqlite_tool import run_sqlite_query


llm = dspy.LM("ollama_chat/phi3.5", api_base="http://localhost:11434")
dspy.settings.configure(lm=llm)
# Get the absolute path to the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the 'docs' directory relative to this script
docs_path = os.path.join(os.path.dirname(script_dir), 'docs')


# --- 1. Define the State ---
class AgentState(TypedDict):
    """
    Represents the state of our LangGraph agent.
    """
    question: str
    format_hint: Optional[str]
    classification: str
    sql_query: str
    sql_results: Optional[List[Any]]
    retrieved_docs: Optional[List[str]]
    errors: List[str]
    retry_count: int
    final_answer: str


# --- 2. Instantiate Tools & Modules ---
retriever = LocalRetriever()
retriever.index_directory(docs_path)

router_module = Router()
sql_generator_module = optimization() # Use the optimized TextToSQL module
synthesizer_module = Synthesizer()


# --- 3. Create Graph Nodes ---

def router_node(state: AgentState):
    """Calls the DSPy Router to classify the question."""
    print("--- ROUTER ---")
    classification = router_module(question=state['question']).classification
    print(f"Classification: {classification}")
    return {"classification": classification, "errors": []}

def retriever_node(state: AgentState):
    """Calls the local retriever to fetch relevant documents."""
    print("--- RETRIEVER ---")
    results = retriever.search(state['question'], k=3)
    # Format for synthesizer: ['filename::chunk_id', 'content']
    formatted_docs = [f"{res['chunk']['source']}::chunk_{res['chunk']['chunk_id']}\n{res['chunk']['content']}\n{res['score']}" for res in results]
    print(f"Retrieved {len(formatted_docs)} documents.")
    return {"retrieved_docs": formatted_docs}

def sql_generator_node(state: AgentState):
    """Generates a SQL query using the DSPy TextToSQL module."""
    print("--- SQL GENERATOR ---")
    # Include previous errors if in a retry loop
    question_with_context = state['question']
    if state.get('errors') and state['errors']:
        error_context = "\n".join(state['errors'])
        question_with_context = f"Previous attempt failed with error: {error_context}. Please correct the query. Original question: {state['question']}"

    response = sql_generator_module(question=question_with_context, schema=schema)
    print(f"Generated SQL: {response.sql_query}")
    return {"sql_query": response.sql_query, "errors": []} # Clear previous errors

def sql_executor_node(state: AgentState):
    """Runs the generated SQL query and captures output or errors."""
    print("--- SQL EXECUTOR ---")
    query = state['sql_query']
    rows, column_names = run_sqlite_query(query)

    if column_names is not None: # Success
        print(f"Execution successful. Rows returned: {len(rows)}")
        # Format results as a list of dictionaries for the synthesizer
        results = [dict(zip(column_names, row)) for row in rows]
        return {"sql_results": results}
    else: # Error
        print(f"Execution failed. Error: {rows}")
        return {"errors": state.get('errors', []) + [rows], "retry_count": state.get('retry_count', 0) + 1}

def synthesizer_node(state: AgentState):
    """Synthesizes the final answer using results from SQL and RAG."""
    print("--- SYNTHESIZER ---")
    context = ""
    if state.get('sql_results'):
        context += f"SQL Results:\n{json.dumps(state.get('sql_results'), indent=2)}\n\n"
    if state.get('retrieved_docs'):
        context += "Retrieved Documents:\n" + "\n\n".join(state.get('retrieved_docs'))

    response = synthesizer_module(
        question=state['question'],
        context=context.strip(),
        sql_query=state.get('sql_query', ''),
        format_hint=state.get('format_hint', 'A clear and concise answer.')
    )
    return {"final_answer": response.json_output}


# --- 4. Define Edges (Conditional Logic) ---

def route_question(state: AgentState):
    """Determines the next step based on the router's classification."""
    print(f"--- DECISION: Based on '{state['classification']}' ---")
    if state['classification'] == 'SQL':
        return "generate_sql"
    elif state['classification'] == 'RAG':
        return "retrieve_docs"
    elif state['classification'] == 'Hybrid':
        return "hybrid_flow"

def check_sql_execution(state: AgentState):
    """Checks for SQL errors and decides whether to retry or synthesize."""
    print(f"--- DECISION: Check SQL execution (Retry count: {state.get('retry_count', 0)}) ---")
    if state.get('errors') and state.get('retry_count', 0) < 2:
        print("SQL Error detected. Retrying generation.")
        return "generate_sql" # Go back to the generator
    print("SQL execution successful or retry limit reached. Proceeding to synthesizer.")
    return "synthesize"


# --- 5. Build the Graph ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router", router_node)
workflow.add_node("retrieve_docs", retriever_node)
workflow.add_node("generate_sql", sql_generator_node)
workflow.add_node("execute_sql", sql_executor_node)
workflow.add_node("synthesize", synthesizer_node)

# Define entry and edges
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_question, { # Corrected method name
    "generate_sql": "generate_sql",
    "retrieve_docs": "retrieve_docs",
    "hybrid_flow": "retrieve_docs" # For hybrid, start with retrieval, then conditionally go to SQL
})

# After retrieval, if the path isn't hybrid, go to synthesize. Otherwise, wait.
workflow.add_edge("retrieve_docs", "synthesize")
# New conditional edge after retrieve_docs to handle RAG-only vs Hybrid
def route_after_retrieval(state: AgentState):
    """Determines the next step after retrieval based on the initial classification."""
    print(f"--- DECISION: After Retrieval (Classification: {state['classification']}) ---")
    if state['classification'] == 'Hybrid':
        return "generate_sql"
    else: # 'RAG'
        return "synthesize"

workflow.add_conditional_edges("retrieve_docs", route_after_retrieval, {
    "generate_sql": "generate_sql",
    "synthesize": "synthesize"
})
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_conditional_edges("execute_sql", check_sql_execution, {
    "generate_sql": "generate_sql", # The repair loop
    "synthesize": "synthesize"
})
workflow.add_edge("synthesize", END)

# Compile the graph
app = workflow.compile()

#from langchain_core.runnables.graph import MermaidDrawMethod

# Assuming 'app' is your LangGraph application instance
#output_file = "langgraph_diagram.png"
#app.get_graph().draw_mermaid_png(
#    output_file_path=output_file,
#    draw_method=MermaidDrawMethod.API,
#    background_color="white",
#    padding=10,
#)
#print(f"Graph saved to {output_file}")