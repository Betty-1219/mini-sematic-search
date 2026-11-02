# 🔍 Mini Semantic Search System

**AWS AASTU AI/ML Training Program — Week 2 Hands-on Assignment**

---

## 📖 Project Description

This project implements a **Mini Semantic Search System** that retrieves FAQ documents based on their meaning rather than relying only on exact keyword matching.

For example, if a user enters:

> How does a machine learn from data?

The system can retrieve relevant FAQs about Machine Learning, even when the exact words in the query do not appear in the original FAQ.

This is achieved using text embeddings, which represent the meaning of text as numerical vectors.

The project uses:

- **Pandas** to load, clean, and prepare FAQ data.
- **Sentence Transformers** to convert text into numerical embeddings.
- **ChromaDB** to store embeddings and retrieve semantically similar documents.

---

## 🎯 Objectives

The main objectives of this project are to:

1. Load and clean structured data using Pandas.
2. Divide text into smaller chunks for better search accuracy.
3. Generate text embeddings using an open-source AI model.
4. Store embeddings in ChromaDB.
5. Implement semantic search based on meaning.
6. Retrieve the top 3 most relevant FAQ results.
7. Build an interactive search system that accepts user queries.

---

## 🧠 Key Concepts

| Concept               | Explanation                                                              |
| --------------------- | ------------------------------------------------------------------------ |
| **Embedding**         | A numerical vector representing the meaning of text.                     |
| **Semantic Search**   | Searching for information based on meaning rather than exact keywords.   |
| **Pandas**            | A Python library for loading and processing structured data.             |
| **ChromaDB**          | A vector database used to store and search embeddings.                   |
| **Vector**            | A list of numerical values representing information.                     |
| **Cosine Similarity** | A mathematical measurement of the similarity between two vectors.        |
| **Chunking**          | Dividing text into smaller sections to improve processing and retrieval. |

---

## 🛠️ Technologies Used

| Technology            | Purpose                     | Version |
| --------------------- | --------------------------- | ------- |
| Python                | Programming language        | 3.9+    |
| Pandas                | Data loading and cleaning   | 2.2.2   |
| Sentence Transformers | Text embedding generation   | 3.0.1   |
| ChromaDB              | Vector database             | 0.5.5   |
| Jupyter Notebook      | Interactive development     | 1.0.0   |
| all-MiniLM-L6-v2      | Pre-trained embedding model | —       |
| Git                   | Version control             | —       |
| GitHub                | Code hosting                | —       |

---

## 📁 Project Structure

```text
mini-sematic-search/
│
├── data/
│   ├── faqs.csv
│   └── chroma_db/ 
│
├── notebooks/
│   └── mini_semantic_search.ipynb
│
├── src/
│   └── semantic_search.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

### Folder Description

- **data/faqs.csv:** Contains 25 FAQ questions and answers across five categories.
- **data/chroma_db/:** Stores the generated vector database. It is created automatically when the application runs.
- **notebooks/:** Contains the Jupyter Notebook implementation.
- **src/:** Contains the main Python implementation.
- **requirements.txt:** Lists the required Python libraries.
- **.gitignore:** Excludes unnecessary files from Git tracking.

---

## ⚙️ Installation and Setup — VS Code

### Step 1: Clone the Repository

Open your VS Code terminal and run:

```bash
git clone https://github.com/Betty-1219/mini-sematic-search.git

cd mini-sematic-search
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

**Windows CMD:**

```cmd
.venv\Scripts\activate.bat
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

The first installation may take several minutes because PyTorch and other dependencies are downloaded.

### Step 5: Run the Python Application

From the project root directory:

```bash
python src/semantic_search.py
```

The application performs the following operations:

1. Loads the FAQ dataset using Pandas.
2. Cleans and prepares the text.
3. Divides text into smaller chunks.
4. Generates embeddings using Sentence Transformers.
5. Stores embeddings in ChromaDB.
6. Executes five example search queries.
7. Displays the top 3 relevant results.
8. Starts an interactive search loop for user questions.

---

## 📓 Running the Jupyter Notebook

Install Jupyter if necessary:

```bash
pip install notebook ipykernel
```

Start Jupyter Notebook:

```bash
python -m jupyter notebook
```

Then:

1. Open `notebooks/mini_semantic_search.ipynb`.
2. Select your Python virtual environment as the kernel.
3. Run the notebook cells from top to bottom.
4. Enter your own questions in the interactive search section.

---

## ☁️ Google Colab Instructions

The project can also be executed using Google Colab without installing the libraries locally.

### Steps

1. Visit [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook**.
3. Upload `notebooks/mini_semantic_search.ipynb`.
4. Upload `data/faqs.csv` when requested by the notebook.
5. Select **Runtime → Run all**.
6. Wait for the required libraries and embedding model to load.
7. Test the system using your own questions.

---

## ▶️ How to Use the System

Run the Python script:

```bash
python src/semantic_search.py
```

The application automatically executes sample queries and then displays:

```text
Interactive semantic search

Type a question, or type 'exit' to stop.

Your question:
```

Enter your question and the system will return the three most relevant FAQ results.

Type `exit` to stop the application.

---

## 🔎 Example Queries

The following queries can be used to test the system:

| Query                                    | Expected Relevant Topic |
| ---------------------------------------- | ----------------------- |
| How does machine learning work?          | Machine Learning        |
| What is a word embedding?                | Embeddings and Search   |
| Cloud file storage on AWS                | Cloud Computing and AWS |
| How do software teams share code?        | Python and Development  |
| How do I store key-value data in Python? | Python and Development  |

The system retrieves relevant information based on semantic meaning, even when the query uses different wording from the original FAQ.

---

## 📊 Expected Output Example

The following is an illustrative example of the expected search output:

```text
================================================================
QUERY: How does machine learning work?
================================================================

Rank #1 | Category: Machine Learning

Similarity: Example score

Question: What is Machine Learning?

Answer:
Machine Learning is a branch of AI that enables computers
to learn patterns from data and improve at a task without
developers writing every rule explicitly.

----------------------------------------------------------------

Rank #2 | Category: Machine Learning

Question: How does a machine learn from data?

Answer:
A machine learning algorithm studies examples, adjusts its
internal parameters to reduce errors, and uses learned
patterns to make predictions on new data.

----------------------------------------------------------------

Rank #3 | Category: Machine Learning

Question: What is supervised learning?

Answer:
Supervised learning trains a model using labeled examples,
where each input is paired with a known correct output.

================================================================
```

**Note:** Actual search results and similarity scores may vary depending on the query, embedding model, and dataset.

### Understanding Similarity Scores

ChromaDB is configured to use cosine distance.

The Python implementation calculates similarity using:

```python
similarity = 1.0 - distance
```

Higher cosine similarity generally indicates that two text embeddings have more similar directions. Similarity scores are not probabilities.

---

## 📦 Git Commit and Push

After completing the project, use one commit to save all project changes and push them to GitHub.

```bash
git add .

git commit -m "Complete mini semantic search system"

git push origin main
```

---

## 📌 Requirements

To run this project, you need:

- Python 3.9 or later.
- VS Code or Jupyter Notebook.
- Git and a GitHub account.
- An internet connection for installing dependencies and downloading the embedding model.
- The required libraries listed in `requirements.txt`.

After downloading the embedding model, the system can potentially run offline if the model files and dependencies are available locally.

---

## 👤 Author

**Name:** Betelhem Nurlachew

**Program:** AWS AASTU AI/ML Training Program

**Week:** 2 — Semantic Search

**GitHub:** [Betty-1219](https://github.com/Betty-1219)

**Project Repository:** [Mini Semantic Search System](https://github.com/Betty-1219/mini-sematic-search)

---

## 📄 License

This project is developed for educational purposes as part of the AWS AASTU AI/ML Training Program.
