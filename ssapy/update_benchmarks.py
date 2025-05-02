import json

def update_documentation():
    # Load benchmark results
    with open("benchmark_results_frame.json", "r") as f:
        results = json.load(f)

    # Format the results for documentation
    formatted_results = "\nBenchmark Results:\n------------------\n"
    for result in results:
        formatted_results += f"Test Name: {result['test_name']}\n"
        formatted_results += f"Execution Time: {result['execution_time']} seconds\n"
        formatted_results += f"Status: {result['status']}\n\n"

    # Append results to benchmarks.rst
    doc_path = "docs/source/benchmarks.rst"
    with open(doc_path, "a") as doc_file:
        doc_file.write(formatted_results)

if __name__ == "__main__":
    update_documentation()