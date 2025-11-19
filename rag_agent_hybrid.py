import argparse
import json
import uuid
import os
from agent.graph_hybrid import app, AgentState

# Get the directory where this script is located to resolve file paths
script_dir = os.path.dirname(os.path.abspath(__file__))

def run_batch(input_file: str, output_file: str):
    """
    Processes a batch of questions from an input JSONL file and writes the
    results to an output JSONL file.

    Args:
        input_file: Path to the input .jsonl file.
        output_file: Path to the output .jsonl file.
    """
    # Construct absolute paths relative to the script's directory
    abs_input_file = os.path.join(script_dir, input_file)
    abs_output_file = os.path.join(script_dir, output_file)

    questions = []
    with open(abs_input_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line))

    with open(abs_output_file, 'w') as f_out:
        for i, item in enumerate(questions):
            print(f"--- Processing Question {i+1}/{len(questions)} ---")
            print(f"Question: {item['question']}")

            # Each conversation gets a unique thread_id for checkpointing
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            # Define the initial state for the graph
            initial_state: AgentState = {
                "question": item["question"],
                "format_hint": item.get("format_hint"),
                "retry_count": 0,
                "errors": [],
                # Initialize other keys to avoid potential KeyErrors in nodes
                "classification": None,
                "sql_query": None,
                "sql_results": None,
                "retrieved_docs": None,
                "final_answer": None,
            }

            # Invoke the LangGraph app
            final_state = app.invoke(initial_state, config=config)

            output_json = {}
            try:
                # The 'final_answer' from the synthesizer is a JSON string, so we parse it
                raw_output = final_state.get("final_answer")
                if raw_output:
                    output_json = json.loads(raw_output)
                else:
                    raise ValueError("Final answer from the agent was empty.")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"!!! ERROR: Failed to parse JSON output for question {i+1}. Writing error to output file. !!!")
                print(f"    Error: {e}")
                print(f"    Model Output: {raw_output}")
                # Create a structured error message for the output file
                output_json = {"id": item.get("id", f"error_{i+1}"), "error": "Failed to generate valid JSON response.", "raw_output": raw_output}

            # Write the structured JSON output to the output file
            f_out.write(json.dumps(output_json) + '\n')
            print(f"--- Finished. Result written to {abs_output_file} ---\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hybrid RAG Agent on a batch of questions.")
    parser.add_argument("--batch", required=True, help="Path to the input JSONL file (e.g., sample_questions_hybrid_eval.jsonl).")
    parser.add_argument("--out", required=True, help="Path to the output JSONL file (e.g., outputs_hybrid.jsonl).")
    args = parser.parse_args()

    run_batch(args.batch, args.out)
