from src.querier import generate_queries, map_answers_to_chunks

# Load the sample markdown file
import json
with open('chunks/sample_md_sliding_window.json', 'r') as f:
    chunks = json.load(f)

# Combine all chunks into one text
full_text = ' '.join(chunk['text'] for chunk in chunks)

doc_id = 'sample_md'
# Generate queries
qa_pairs = generate_queries(doc_id, full_text, num_qs=3)

# Print the raw response first
print("\nRaw Response:")
print("============")
print(qa_pairs)

# Print the formatted results
print("\nGenerated Question-Answer Pairs:")
print("================================")
if isinstance(qa_pairs, dict) and "questions" in qa_pairs:
    questions = qa_pairs["questions"]
    for i, qa in enumerate(questions, 1):
        print(f"\n{i}. Question: {qa['question']}")
        print(f"   Answer: {qa['answer']}")
else:
    print("Unexpected response format:", type(qa_pairs))
    questions = []

# Map answers to chunks
mapped = map_answers_to_chunks(doc_id, questions, 'chunks')

# Save the mapped output as JSON in gold_queries
output_path = 'gold_queries/sample_md_mapped.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(mapped, f, indent=2, ensure_ascii=False)
print(f"\nMapped answers saved to {output_path}") 