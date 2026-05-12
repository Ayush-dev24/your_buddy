import os
import chromadb
from sentence_transformers import CrossEncoder

def main():
    print("Testing ChromaDB and Sentence-Transformers imports...")
    try:
        # Test Sentence-Transformers
        print("Loading CrossEncoder model (this may take a moment the first time)...")
        re_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("CrossEncoder loaded successfully!")
        
        # Test ChromaDB
        print("Initializing ChromaDB PersistentClient...")
        db_path = os.path.join(os.path.dirname(__file__), "chroma_db_test")
        client = chromadb.PersistentClient(path=db_path)
        
        collection = client.get_or_create_collection(name="test_collection")
        print("ChromaDB collection created/retrieved successfully!")
        
        # Add some dummy data
        collection.add(
            documents=["This is a test document about AI.", "Another document about cats."],
            metadatas=[{"topic": "AI"}, {"topic": "animals"}],
            ids=["doc1", "doc2"]
        )
        print("Dummy documents added successfully!")
        
        # Query
        results = collection.query(
            query_texts=["Tell me about artificial intelligence"],
            n_results=1
        )
        print("Query results:", results)
        
        # Test Re-ranking
        pairs = [("Tell me about artificial intelligence", "This is a test document about AI.")]
        scores = re_ranker.predict(pairs)
        print("Re-ranking score for correct pair:", scores[0])
        
        print("\nAll tests passed successfully!")
    except Exception as e:
        print("Error during test:", str(e))

if __name__ == "__main__":
    main()
