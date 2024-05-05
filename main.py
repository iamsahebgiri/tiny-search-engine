from flask import Flask, render_template, request, send_from_directory
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import markdown

app = Flask(
    __name__,
)


def read_markdown_files(directory):
    markdown_files = glob.glob(os.path.join(directory, "**/*.md"))
    markdown_data = []
    for file_path in markdown_files:
        with open(file_path, "r", encoding="utf-8") as file:
            markdown_content = file.read()
            markdown_data.append({"path": file_path, "content": markdown_content})
    return markdown_data


def search_markdown_files(markdown_data, search_query, top_n=10):
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    documents = [item["content"] for item in markdown_data]
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    query_vector = tfidf_vectorizer.transform([search_query])

    cosine_similarities = np.dot(query_vector, tfidf_matrix.T)
    cosine_similarities = cosine_similarities.toarray().flatten()

    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    top_results = [
        {"path": markdown_data[i]["path"], "score": cosine_similarities[i]}
        for i in top_indices
    ]
    return top_results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["GET", "POST"])
def search():
    search_query = request.args.get("query") or request.form.get("query")
    directory = "./docs"

    markdown_data = read_markdown_files(directory)
    search_results = search_markdown_files(markdown_data, search_query)

    return render_template("search_results.html", results=search_results)


@app.route("/view/<path:file_path>", methods=["GET"])
def view(file_path):
    if file_path.endswith(".md") and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            markdown_content = file.read()
            html_content = markdown.markdown(
                markdown_content,
                extensions=[
                    "pymdownx.arithmatex",
                    "pymdownx.highlight",
                    "admonition",
                    "pymdownx.details",
                    "pymdownx.superfences",
                    "pymdownx.tabbed",
                    "attr_list",
                    "pymdownx.emoji",
                    "meta",
                    "toc",
                ],
                extension_configs={
                    "pymdownx.arithmatex": {
                        "generic": True,
                        "tex_inline_wrap": ["$", "$"],
                        "tex_block_wrap": ["$$", "$$"],
                    },
                    "pymdownx.highlight": {"use_pygments": False},
                    "pymdownx.tabbed": {"alternate_style": True},
                    "toc": {"permalink": True},
                },
            )
        return render_template("view.html", content=html_content)
    elif file_path.endswith(".png") or file_path.endswith(".jpg"):
        print(file_path)
        return send_from_directory("", file_path)
    else:
        return "Invalid file path"


if __name__ == "__main__":
    app.run(debug=True)
