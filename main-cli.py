import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def read_markdown_files(directory):
    markdown_files = glob.glob(os.path.join(directory, "**/*.md"))
    markdown_data = []
    for file_path in markdown_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            markdown_content = file.read()
            markdown_data.append({'path': file_path, 'content': markdown_content})
    return markdown_data

def search_markdown_files(markdown_data, search_query, top_n=10):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    documents = [item['content'] for item in markdown_data]
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    query_vector = tfidf_vectorizer.transform([search_query])

    cosine_similarities = np.dot(query_vector, tfidf_matrix.T)
    cosine_similarities = cosine_similarities.toarray().flatten()

    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    top_results = [{'path': markdown_data[i]['path'], 'score': cosine_similarities[i]} for i in top_indices]
    return top_results

if __name__ == "__main__":
    directory = "./docs"
    search_query = input("Enter search query: ")

    markdown_data = read_markdown_files(directory)
    # print("Number of markdown files:", len(markdown_data))
    search_results = search_markdown_files(markdown_data, search_query)
    
    print("_"*50)
    print("\nTop 10 search results:")
    for result in search_results:
        print("Path:", result['path'])
        print("Score:", result['score'])
        print()
