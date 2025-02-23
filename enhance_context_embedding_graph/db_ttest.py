import os
import voyageai
from pymilvus import MilvusClient


class SimpleVectorDB:
    def __init__(self, collection_name: str):
        self.voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.milvus_client = MilvusClient(uri="http://localhost:19530")  # Fixed URI format
        self.collection_name = collection_name
        self.dimension = 1024

    def setup_collection(self):
        """Create or recreate the collection"""
        # Drop if exists
        if self.milvus_client.has_collection(collection_name=self.collection_name):
            print(f"Dropping existing collection {self.collection_name}")
            self.milvus_client.drop_collection(collection_name=self.collection_name)

        # Create new collection
        print(f"Creating new collection {self.collection_name}")
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension
        )

    def insert_documents(self, texts: list[str]):
        """Insert documents into the collection"""
        print("\nGenerating embeddings...")
        vectors = self.voyage_client.embed(
            texts=texts,
            model="voyage-2",
            truncation=False
        ).embeddings

        print("\nPreparing data for insertion...")
        # Print debug info about vectors
        print(f"Vector type: {type(vectors[0])}")
        print(f"Vector length: {len(vectors[0])}")

        data = [
            {"id": i, "vector": vectors[i], "text": texts[i]}
            for i in range(len(texts))
        ]

        print("\nInserting data...")
        result = self.milvus_client.insert(
            collection_name=self.collection_name,
            data=data
        )
        print(f"Inserted {result['insert_count']} documents")

    def search(self, query: str, limit: int = 2):
        """Search for similar documents"""
        print(f"\nSearching for: {query}")

        # Generate query vector
        query_vector = self.voyage_client.embed(
            texts=[query],
            model="voyage-2",
            truncation=False
        ).embeddings[0]  # Take first vector since we only have one query

        # Search
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_vector],  # Wrap in list as search expects list of vectors
            limit=limit,
            output_fields=["text"]
        )

        return results


def main():
    # Initialize database
    db = SimpleVectorDB("test_collection")

    # Setup collection
    print("Setting up collection...")
    db.setup_collection()

    # Test data
    test_docs = [
        "The quick brown fox jumps over the lazy dog.",
        "A software engineer writes code for computer programs.",
        "Machine learning models learn patterns from data.",
    ]

    # Insert documents
    db.insert_documents(test_docs)

    # Test search
    search_query = "How do computers learn?"
    results = db.search(search_query)

    print("\nSearch results:")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()