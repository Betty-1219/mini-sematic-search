"""
=============================================================
  Mini Semantic Search System
  AWS AASTU AI/ML Training Program – Week 2
=============================================================

WHAT THIS FILE DOES (plain English):
--------------------------------------
This file is the brain of the project. It contains all the
functions needed to build and run a semantic search system:

  1. Load FAQ data from a CSV file using Pandas
  2. Clean the text (remove extra spaces, bad characters)
  3. Split long answers into smaller chunks
  4. Convert text chunks into embeddings (lists of numbers
     that capture the MEANING of the text)
  5. Store embeddings in ChromaDB (a vector database)
  6. Search ChromaDB by meaning when the user asks a question
  7. Display the top 3 most relevant FAQ results

KEY CONCEPTS FOR BEGINNERS:
--------------------------------------
- EMBEDDING: A list of numbers (a "vector") that represents
  the meaning of a piece of text. Similar texts will have
  similar vectors. Example: "dog" and "puppy" will be close
  together in this number-space.

- SEMANTIC SEARCH: Finding results based on meaning, not
  just exact word matching. You can search "how does ML work"
  and get results about "machine learning explained" even if
  those exact words don't match.

- PANDAS: A Python library for working with data tables
  (like spreadsheets). It loads CSV files into a structure
  called a DataFrame — rows and columns you can filter,
  clean, and analyse with Python code.

- CHROMADB: A vector database. It stores embeddings and can
  find the ones that are closest in meaning to a query
  embedding. This is what makes semantic search fast.
"""

# =============================================================
# STAGE 1: IMPORT DEPENDENCIES
# =============================================================
# We import the tools (libraries) we need.
# Think of imports like getting your tools out of the toolbox
# before you start working.

import os       # Used to build file paths that work on any OS
import re       # Regular expressions: a tool for cleaning text

import pandas as pd                           # Table/spreadsheet operations
import chromadb                               # Vector database
from sentence_transformers import SentenceTransformer  # Creates embeddings


# =============================================================
# CONFIGURATION CONSTANTS
# =============================================================
# These are settings we use throughout the file.
# Putting them here at the top makes them easy to change.

COLLECTION_NAME = "faq_collection"   # Name for our ChromaDB collection
CHUNK_SIZE      = 500                # Max characters per text chunk
CHUNK_OVERLAP   = 50                 # Characters shared between chunks
TOP_K           = 3                  # How many results to return
MODEL_NAME      = "all-MiniLM-L6-v2" # The free, open-source embedding model


# =============================================================
# STAGE 2: LOAD FAQ DATA
# =============================================================

def load_faq_data(csv_path: str) -> pd.DataFrame:
    """
    Load the FAQ CSV file into a Pandas DataFrame.

    WHY PANDAS?
    -----------
    A DataFrame is like a table with rows (each FAQ) and
    columns (id, category, question, answer). Pandas makes
    it very easy to:
      - Read a CSV file in one line of code
      - Check for missing values
      - Filter and clean rows
      - Apply functions to every row at once

    Parameters
    ----------
    csv_path : str
        The full path to the faqs.csv file.

    Returns
    -------
    pd.DataFrame
        A table containing all FAQ rows.
    """
    print("\n" + "=" * 55)
    print("  STAGE 2: Loading FAQ Data")
    print("=" * 55)
    print(f"  Path: {csv_path}")

    # Check the file exists before trying to open it
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n❌ File not found: '{csv_path}'\n"
            "Please make sure faqs.csv is inside the data/ folder."
        )

    # pd.read_csv() reads the CSV file and returns a DataFrame
    df = pd.read_csv(csv_path)

    print(f"\n  ✅ Loaded successfully!")
    print(f"     Rows    : {len(df)}")
    print(f"     Columns : {list(df.columns)}")

    return df


# =============================================================
# STAGE 3: INSPECT AND CLEAN THE DATA
# =============================================================

def clean_text(text: str) -> str:
    """
    Clean a single piece of text.

    Steps performed (in order):
      1. Convert to string (handles numbers or None values)
      2. Strip leading/trailing whitespace
      3. Replace newlines and tabs with a single space
      4. Collapse multiple spaces into one space
      5. Remove non-printable/non-ASCII characters

    Parameters
    ----------
    text : str
        The raw text string to clean.

    Returns
    -------
    str
        The cleaned text string.
    """
    # Make sure it is a string
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()                          # Remove outer whitespace
    text = re.sub(r"[\n\r\t]+", " ", text)       # Newlines → space
    text = re.sub(r" {2,}", " ", text)           # Multiple spaces → one
    text = re.sub(r"[^\x20-\x7E]+", "", text)   # Remove non-ASCII chars
    text = text.strip()

    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inspect the data and apply cleaning to the whole DataFrame.

    Steps:
      1. Print a preview of the raw data (first 3 rows)
      2. Show which columns have missing values
      3. Drop rows where 'question' or 'answer' is missing
      4. Apply clean_text() to every question and answer
      5. Drop any row where question or answer is now empty
      6. Reset the row numbering

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame from load_faq_data().

    Returns
    -------
    pd.DataFrame
        A clean, ready-to-use DataFrame.
    """
    print("\n" + "=" * 55)
    print("  STAGE 3: Inspecting and Cleaning Data")
    print("=" * 55)

    # Show the first 3 rows so we can see the data structure
    print("\n  First 3 rows of raw data:")
    print(df.head(3).to_string())

    # Show missing value counts per column
    print(f"\n  Missing values per column:")
    print(df.isnull().sum().to_string())

    # Drop rows where question or answer is completely missing
    before = len(df)
    df = df.dropna(subset=["question", "answer"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"\n  🗑️  Dropped {dropped} rows with missing values.")

    # Apply clean_text() to every cell in the question and answer columns
    # .apply() runs a function on every row automatically
    df["question"] = df["question"].apply(clean_text)
    df["answer"]   = df["answer"].apply(clean_text)

    # Drop rows where cleaning left us with an empty string
    df = df[df["question"].str.strip() != ""]
    df = df[df["answer"].str.strip()   != ""]

    # Reset the index (row numbers) so they start from 0 again
    df = df.reset_index(drop=True)

    print(f"\n  ✅ Clean dataset: {len(df)} rows ready.")
    return df


# =============================================================
# STAGE 4: SPLIT LONG ANSWERS INTO CHUNKS
# =============================================================

def split_text_into_chunks(text: str,
                           chunk_size: int = CHUNK_SIZE,
                           overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split a long text string into smaller overlapping chunks.

    WHY DO WE CHUNK?
    ----------------
    Embedding models work best on shorter texts (under ~512
    tokens / ~400 words). Very long answers can lose detail
    when squeezed into a single embedding. Chunking gives
    each part of a long answer its own embedding, making
    search more precise.

    WHY OVERLAP?
    ------------
    If we split at character 500 exactly, we might cut a
    sentence in half. Using overlap means the beginning of
    the next chunk repeats the end of the previous one, so
    no sentence is ever lost at a boundary.

    Example with chunk_size=20, overlap=5 on "ABCDEFGHIJKLMNOP":
      Chunk 1: ABCDEFGHIJKLMNOP (chars 0-19)
      Chunk 2: LMNOP...         (starts at char 15 = 20-5)

    Parameters
    ----------
    text       : str   The full text to split.
    chunk_size : int   Maximum characters per chunk.
    overlap    : int   Characters repeated between chunks.

    Returns
    -------
    list of str
        A list of text chunks.
    """
    chunks = []
    start  = 0

    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:               # Only add non-empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap   # Advance with overlap

    return chunks


def create_all_chunks(df: pd.DataFrame) -> list:
    """
    Process every FAQ row and produce a list of chunk records.

    For each row we:
      1. Combine the question + answer into one text block
         (including the question in the chunk helps the
          embedding capture the full context of that FAQ)
      2. Split into chunks using split_text_into_chunks()
      3. Build a record (dictionary) for each chunk that
         stores the chunk text plus all the FAQ metadata

    Each record looks like this:
      {
        "chunk_id"  : "faq_3_chunk_1",   ← unique ID
        "chunk_text": "Question: ... Answer: ...",
        "question"  : "What is Python?",
        "answer"    : "Python is ...",
        "category"  : "Python",
        "faq_id"    : "3"
      }

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned DataFrame.

    Returns
    -------
    list of dict
        Every chunk record across all FAQ rows.
    """
    print("\n" + "=" * 55)
    print("  STAGE 4: Splitting Answers into Chunks")
    print("=" * 55)

    all_chunks = []

    for idx, row in df.iterrows():
        # Build the text we want to embed for this FAQ row
        combined_text = (
            f"Question: {row['question']} "
            f"Answer: {row['answer']}"
        )

        # Split into chunks
        chunks = split_text_into_chunks(combined_text)

        # Build a record for each chunk
        for chunk_idx, chunk_text in enumerate(chunks):
            record = {
                "chunk_id"  : f"faq_{idx}_chunk_{chunk_idx}",
                "chunk_text": chunk_text,
                "question"  : row["question"],
                "answer"    : row["answer"],
                "category"  : str(row.get("category", "General")),
                "faq_id"    : str(row.get("id", idx)),
            }
            all_chunks.append(record)

    print(f"\n  ✅ Created {len(all_chunks)} chunks from {len(df)} FAQ rows.")
    print(f"     Average chunks per FAQ: "
          f"{len(all_chunks) / max(len(df), 1):.1f}")
    return all_chunks


# =============================================================
# STAGE 5: GENERATE EMBEDDINGS
# =============================================================

def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Download and load the Sentence Transformer embedding model.

    WHAT IS AN EMBEDDING MODEL?
    ----------------------------
    This model reads a sentence and outputs a list of 384
    numbers (a vector). The numbers encode the meaning.
    Sentences with similar meaning get similar numbers.

    We use 'all-MiniLM-L6-v2' because it is:
      - Free and open source (no API key needed)
      - Small and fast (runs on CPU)
      - Accurate enough for our FAQ search task

    The first time you run this it will download the model
    (~90 MB) from Hugging Face. After that it is cached
    locally and loads instantly.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.

    Returns
    -------
    SentenceTransformer
        The loaded model object.
    """
    print("\n" + "=" * 55)
    print("  STAGE 5: Loading Embedding Model")
    print("=" * 55)
    print(f"\n  Model : {model_name}")
    print("  (Downloads ~90 MB on first run, then cached)")

    model = SentenceTransformer(model_name)

    print("  ✅ Embedding model loaded.")
    return model


def generate_embeddings(chunks: list,
                        model: SentenceTransformer) -> list:
    """
    Generate an embedding vector for every chunk.

    We pass ALL chunk texts to model.encode() at once.
    The model processes them in batches internally, which
    is faster than encoding one by one.

    The result is a 2D array of shape:
        (number_of_chunks, 384)
    — one 384-number vector per chunk.

    We then attach each vector to its chunk record as
    the 'embedding' key (converted to a Python list
    because ChromaDB requires plain Python lists).

    Parameters
    ----------
    chunks : list of dict
        Chunk records from create_all_chunks().
    model  : SentenceTransformer
        The loaded embedding model.

    Returns
    -------
    list of dict
        Same chunk records, each now with 'embedding' key.
    """
    print("\n" + "=" * 55)
    print("  Generating Embeddings")
    print("=" * 55)
    print(f"\n  Encoding {len(chunks)} chunks... (progress bar below)")

    # Extract just the text strings
    texts = [chunk["chunk_text"] for chunk in chunks]

    # Encode all texts; show_progress_bar gives a live % bar
    embeddings = model.encode(texts, show_progress_bar=True)

    # Attach embedding to each chunk record
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()  # numpy → Python list

    print(f"\n  ✅ Embeddings generated. Each vector has "
          f"{len(chunks[0]['embedding'])} dimensions.")
    return chunks


# =============================================================
# STAGE 6: CREATE CHROMADB COLLECTION
# =============================================================

def setup_chromadb(persist_directory: str,
                   collection_name: str = COLLECTION_NAME):
    """
    Connect to ChromaDB and create (or reuse) a collection.

    WHAT IS CHROMADB?
    -----------------
    ChromaDB is a vector database — a special database that
    stores embeddings (lists of numbers) and can find the
    ones that are closest to a query embedding.

    Unlike a regular database that searches by matching text
    exactly, ChromaDB searches by mathematical distance
    between vectors. Closer distance = more similar meaning.

    We use 'cosine' similarity:
      - Score of 1.0 = perfectly identical meaning
      - Score of 0.0 = completely unrelated

    PersistentClient saves the database to disk so your data
    survives between runs. You only need to embed and store
    once; after that ChromaDB reloads from disk.

    Parameters
    ----------
    persist_directory : str
        Folder path where ChromaDB will save its files.
    collection_name   : str
        Name for this collection (like a table name).

    Returns
    -------
    tuple : (client, collection)
    """
    print("\n" + "=" * 55)
    print("  STAGE 6: Setting Up ChromaDB")
    print("=" * 55)
    print(f"\n  Storage folder : {persist_directory}")
    print(f"  Collection name: {collection_name}")

    # Create the folder if it does not already exist
    os.makedirs(persist_directory, exist_ok=True)

    # Connect to (or create) the persistent ChromaDB database
    client = chromadb.PersistentClient(path=persist_directory)

    # get_or_create_collection: safe to call multiple times
    # If collection exists already it just returns it
    collection = client.get_or_create_collection(
        name     = collection_name,
        metadata = {"hnsw:space": "cosine"}  # cosine distance metric
    )

    print(f"\n  ✅ Collection ready. "
          f"Documents already stored: {collection.count()}")
    return client, collection


# =============================================================
# STAGE 7: STORE EMBEDDINGS AND METADATA
# =============================================================

def store_chunks_in_chromadb(collection, chunks: list) -> None:
    """
    Store text chunks, embeddings, and metadata in ChromaDB.

    WHAT GETS STORED?
    -----------------
    For each chunk we store three things:
      1. id        — unique string identifier for the chunk
      2. embedding — the 384-number vector (the meaning)
      3. document  — the raw text of the chunk
      4. metadata  — extra information: original question,
                     full answer, category, faq_id

    Later, when we search, ChromaDB returns all of this
    information so we can display the original FAQ to
    the user.

    DUPLICATE PREVENTION:
    ---------------------
    We check which IDs already exist and skip re-adding
    them. This makes re-running the notebook safe — it
    won't store duplicates.

    Parameters
    ----------
    collection : chromadb.Collection
        The ChromaDB collection to add data to.
    chunks     : list of dict
        Chunk records that already have 'embedding' keys.
    """
    print("\n" + "=" * 55)
    print("  STAGE 7: Storing Data in ChromaDB")
    print("=" * 55)

    # Get IDs already stored so we can skip duplicates
    existing_ids = set(collection.get()["ids"])
    print(f"\n  Already stored: {len(existing_ids)} items")

    # Filter to only new chunks
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    print(f"  New chunks to add: {len(new_chunks)}")

    if not new_chunks:
        print("\n  ℹ️  Nothing new to store — all chunks already in ChromaDB.")
        return

    # Build the lists ChromaDB expects
    ids        = [c["chunk_id"]   for c in new_chunks]
    embeddings = [c["embedding"]  for c in new_chunks]
    documents  = [c["chunk_text"] for c in new_chunks]
    metadatas  = [
        {
            "question": c["question"],
            "answer"  : c["answer"],
            "category": c["category"],
            "faq_id"  : c["faq_id"],
        }
        for c in new_chunks
    ]

    # Add in batches of 50 to avoid memory issues
    batch_size = 50
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids        = ids[start:end],
            embeddings = embeddings[start:end],
            documents  = documents[start:end],
            metadatas  = metadatas[start:end],
        )

    total = collection.count()
    print(f"\n  ✅ Stored {len(new_chunks)} new chunks.")
    print(f"     Total items in ChromaDB: {total}")


# =============================================================
# STAGE 8: SEMANTIC SEARCH
# =============================================================

def semantic_search(query: str,
                    collection,
                    model: SentenceTransformer,
                    top_k: int = TOP_K) -> list:
    """
    Search the ChromaDB collection for the best FAQ matches.

    HOW SEMANTIC SEARCH WORKS (step by step):
    ------------------------------------------
    Step 1: Take the user's question (a string of text)
    Step 2: Convert it into an embedding using the SAME
            model we used to embed the FAQ chunks.
            → This gives us a 384-number query vector.
    Step 3: Send that vector to ChromaDB.
    Step 4: ChromaDB computes the cosine distance between
            the query vector and every stored vector.
    Step 5: It returns the top_k chunks whose vectors are
            CLOSEST (most similar in meaning) to the query.
    Step 6: We format the results and return them.

    WHY DOES THIS FIND RELATED MEANING?
    ------------------------------------
    Because the same model was used for both the FAQ text
    and the user query, texts with similar meaning land in
    similar regions of the 384-dimensional number-space.
    ChromaDB just finds the nearest neighbours.

    Parameters
    ----------
    query      : str   The user's question.
    collection : chromadb.Collection  Where to search.
    model      : SentenceTransformer  For embedding the query.
    top_k      : int   Number of results to return.

    Returns
    -------
    list of dict
        Each result has: rank, question, answer, category,
        similarity_score, chunk_text.
    """
    if not query.strip():
        print("  ⚠️  Empty query. Please enter a question.")
        return []

    # Step 1: Embed the user query
    query_embedding = model.encode(query).tolist()

    # Step 2: Query ChromaDB for nearest neighbours
    raw_results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = top_k,
        include          = ["metadatas", "documents", "distances"]
    )

    # Step 3: Format results into clean dictionaries
    results   = []
    metadatas = raw_results["metadatas"][0]   # list of metadata dicts
    distances = raw_results["distances"][0]   # cosine distances (0=closest)
    documents = raw_results["documents"][0]   # the chunk text strings

    for rank, (meta, dist, doc) in enumerate(
            zip(metadatas, distances, documents), start=1):

        # Convert distance to similarity score
        # cosine distance: 0 = identical, 1 = opposite
        # cosine similarity: 1 = identical, 0 = opposite
        similarity = round(1.0 - dist, 4)

        results.append({
            "rank"            : rank,
            "question"        : meta.get("question", "N/A"),
            "answer"          : meta.get("answer",   "N/A"),
            "category"        : meta.get("category", "N/A"),
            "similarity_score": similarity,
            "chunk_text"      : doc,
        })

    return results


# =============================================================
# STAGE 9 & 10: DISPLAY TOP 3 RESULTS
# =============================================================

def display_results(query: str, results: list) -> None:
    """
    Print the top 3 search results in a clear, readable format.

    For each result we show:
      - Rank number
      - Category
      - Similarity score (0.0 to 1.0) with a visual bar
      - The original FAQ question
      - The first 300 characters of the answer

    Parameters
    ----------
    query   : str        The original user query.
    results : list       Results from semantic_search().
    """
    print("\n" + "=" * 60)
    print(f"  🔎  QUERY: {query}")
    print("=" * 60)

    if not results:
        print("  ❌  No results found.")
        return

    for r in results:
        # Build a simple visual bar from the similarity score
        filled = int(r["similarity_score"] * 20)
        bar    = "█" * filled + "░" * (20 - filled)

        # Truncate long answers for display
        answer_preview = r["answer"][:300]
        if len(r["answer"]) > 300:
            answer_preview += "..."

        print(f"\n  ┌─ Rank #{r['rank']}  |  Category: {r['category']}")
        print(f"  │  Similarity : {r['similarity_score']:.4f}  [{bar}]")
        print(f"  │")
        print(f"  │  ❓ Question : {r['question']}")
        print(f"  │  💡 Answer   : {answer_preview}")
        print(f"  └" + "─" * 55)

    print()


# =============================================================
# FULL PIPELINE FUNCTION
# =============================================================

def build_search_index(csv_path: str, chroma_dir: str):
    """
    Run the complete pipeline from CSV to searchable index.

    Stages run in order:
      2. Load data  →  3. Clean  →  4. Chunk  →
      5. Embed      →  6. Setup ChromaDB  →  7. Store

    Safe to call multiple times: existing data is detected
    and skipped so you won't get duplicates.

    Parameters
    ----------
    csv_path  : str  Path to faqs.csv
    chroma_dir: str  Directory for ChromaDB persistence

    Returns
    -------
    tuple : (collection, model)
        Ready to be used with semantic_search()
    """
    df         = load_faq_data(csv_path)
    df         = clean_dataframe(df)
    chunks     = create_all_chunks(df)
    model      = load_embedding_model()
    chunks     = generate_embeddings(chunks, model)
    _, collection = setup_chromadb(chroma_dir)
    store_chunks_in_chromadb(collection, chunks)

    return collection, model


# =============================================================
# INTERACTIVE TERMINAL MODE
# =============================================================

def run_interactive_search(collection, model: SentenceTransformer) -> None:
    """
    Start an interactive question-answering loop in the terminal.

    The user types a question → results are displayed.
    Type 'quit' or 'exit' or press Ctrl+C to stop.

    Parameters
    ----------
    collection : chromadb.Collection
    model      : SentenceTransformer
    """
    print("\n" + "=" * 60)
    print("  🚀  Mini Semantic Search  —  AWS AASTU AI/ML Week 2")
    print("=" * 60)
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("  ❓ Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  👋 Exiting. Goodbye!")
            break

        if not query:
            print("  ⚠️  Please type something before pressing Enter.\n")
            continue

        if query.lower() in {"quit", "exit", "q", "bye"}:
            print("  👋 Goodbye!")
            break

        results = semantic_search(query, collection, model)
        display_results(query, results)


# =============================================================
# ENTRY POINT — run as a script: python src/semantic_search.py
# =============================================================

if __name__ == "__main__":

    # Build paths relative to this file's location
    # This makes it work whether you run from the project root
    # or from inside the src/ folder
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH   = os.path.join(BASE_DIR, "data", "faqs.csv")
    CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

    print("\n  Welcome to the Mini Semantic Search System!")
    print("  Building the search index...\n")

    # Build (or reload) the index
    collection, model = build_search_index(CSV_PATH, CHROMA_DIR)

    # --- STAGE 9: Run 5 built-in test queries ---
    test_queries = [
        "How does machine learning actually work?",
        "What is a word embedding?",
        "Tell me about cloud file storage on AWS",
        "How do I organise data in Python?",
        "What is the difference between AI and machine learning?",
        "How do software teams collaborate on code?",
    ]

    print("\n\n" + "=" * 60)
    print("  🧪  STAGE 9: RUNNING EXAMPLE TEST QUERIES")
    print("=" * 60)

    for q in test_queries:
        results = semantic_search(q, collection, model)
        display_results(q, results)

    # --- Start interactive mode ---
    run_interactive_search(collection, model)
