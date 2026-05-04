import os
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Clients
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("codebase")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1 — Load your documents
documents = [
    {
        "id": "doc1",
        "text": """
        def review_code(code_to_review, retries=3):
            for attempt in range(retries):
                try:
                    response = anthropic_client.messages.create(...)
                    return response.content[0].text
                except anthropic.AuthenticationError:
                    print("Invalid API key")
                    raise
                except anthropic.APIError as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise
        """,
        "source": "code_review_agent.py - error handling"
    },
    {
        "id": "doc2",
        "text": """
        TEST_CASES = [
            {"name": "SQL Injection", "code": "...", "must_catch": ["sql injection"]},
            {"name": "Hardcoded Password", "code": "...", "must_catch": ["password"]},
            {"name": "Division By Zero", "code": "...", "must_catch": ["division"]},
            {"name": "Resource Leak", "code": "...", "must_catch": ["close"]},
        ]
        def run_evals():
            for test in TEST_CASES:
                review = run_review(test["code"]).lower()
                caught = any(term in review for term in test["must_catch"])
        """,
        "source": "evals.py - evaluation framework"
    },
    {
        "id": "doc3",
        "text": """
        def get_pr_diff(repo_name, pr_number):
            try:
                repo = github_client.get_repo(repo_name)
                pr = repo.get_pull(pr_number)
                code_to_review = ""
                for file in pr.get_files():
                    if file.patch:
                        code_to_review += f"### File: {file.filename}"
                        code_to_review += file.patch
                return pr, code_to_review
            except GithubException as e:
                print(f"GitHub error: {e.status}")
                raise
        """,
        "source": "code_review_agent.py - GitHub integration"
    },
    {
        "id": "doc4",
        "text": """
        def post_review(pr, review):
            try:
                pr.create_issue_comment(review)
                print("Review posted to GitHub PR!")
            except GithubException as e:
                print(f"Failed to post review: {e.status}")
                raise
        """,
        "source": "code_review_agent.py - posting reviews"
    },
]

# Step 2 — Convert documents to embeddings and store
print("Indexing documents...")
for doc in documents:
    embedding = embedder.encode(doc["text"]).tolist()
    collection.add(
        ids=[doc["id"]],
        embeddings=[embedding],
        documents=[doc["text"]],
        metadatas=[{"source": doc["source"]}]
    )
print(f"Indexed {len(documents)} documents\n")

# Step 3 — Answer questions using RAG
def ask(question):
    # Convert question to embedding
    question_embedding = embedder.encode(question).tolist()

    # Find most relevant documents
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=2
    )

    # Build context from retrieved docs
    context = ""
    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]
        context += f"\n### {source}\n{doc}\n"

    print(f"Retrieved context from: {[m['source'] for m in results['metadatas'][0]]}\n")

    # Send to Claude with context
    response = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system="You are a helpful assistant that answers questions about a codebase. Use only the provided context to answer.",
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    return response.content[0].text

# Test it
questions = [
    "How does error handling work?",
    "How are reviews posted to GitHub?",
    "What does the evals framework test for?",
]

for q in questions:
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print(f"{'='*60}")
        answer = ask(q)
        print(f"\nAnswer:\n{answer}")
        print()