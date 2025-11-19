import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class LocalRetriever:
    """
    A local retriever using TF-IDF to find relevant chunks from Markdown files.
    """
    def __init__(self):
        self.chunks = []
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 2), 
            max_df=0.85, 
            min_df=2
        )
        self.tfidf_matrix = None

    def _split_into_chunks(self, content, filename):
        """Splits markdown content into paragraph chunks."""
        # Split by two or more newlines to define paragraphs
        paragraphs = re.split(r'\n\n+', content)
        for i, para in enumerate(paragraphs):
            if para.strip(): # Only add non-empty paragraphs
                self.chunks.append({
                    'content': para.strip(),
                    'source': filename,
                    'chunk_id': i
                })

    def index_directory(self, path):
        """
        Walks through a directory, splits markdown files into chunks, and creates a TF-IDF index.
        
        Args:
            path: The path to the directory containing markdown files.
        """
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".md"):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self._split_into_chunks(f.read(), file)
        
        if self.chunks:
            chunk_contents = [chunk['content'] for chunk in self.chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(chunk_contents)

    def search(self, query, k=5):
        """
        Searches the indexed chunks for the most relevant ones to the query.

        Args:
            query: The search query string.
            k: The number of top chunks to return.

        Returns:
            A list of dictionaries, where each dictionary contains a chunk 
            and its relevance score. Returns an empty list if not indexed.
        """
        if self.tfidf_matrix is None or not self.chunks:
            print("No documents have been indexed. Please call index_directory() first.")
            return []

        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between the query and all chunks
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get the indices of the top k chunks
        # np.argsort returns indices of the sorted array in ascending order
        # We use [-k:] to get the top k and then [::-1] to reverse for descending order
        top_k_indices = np.argsort(cosine_similarities)[-k:][::-1]
        
        results = []
        for i in top_k_indices:
            results.append({
                'score': cosine_similarities[i],
                'chunk': self.chunks[i]
            })
        return results