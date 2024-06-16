import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import re
import nltk
import webbrowser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the main window
root = tk.Tk()
root.title("Information Retrieval System")
root.geometry("800x600")

# Global variables
data = None
corpus = []

def load_file():
    global data, corpus
    data = None
    corpus = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(script_dir, "dataset.csv")
    
    try:
        data = pd.read_csv(file, encoding='latin-1')
        if len(data.columns) >= 2:
            data.columns = ['label', 'text', 'penulis', 'tahun', 'link']
        else:
            raise ValueError("CSV file must have at least two columns")
        preprocess_text()
    except Exception as e:
        messagebox.showerror("Error", f"Error loading file: {str(e)}")

# Function to preprocess text
def preprocess_text():
    global corpus
    text = list(data['text'])
    lemmatizer = WordNetLemmatizer()
    for i in range(len(text)):
        r = re.sub('[^a-zA-Z]', ' ', text[i])
        r = re.sub('[!@#$%^&*()_+-<>?/.:;"{}’[]|\]', ' ', text[i])
        r = r.lower()
        r = r.split()
        r = [word for word in r if word not in stopwords.words('english')]
        r = [lemmatizer.lemmatize(word) for word in r]
        r = ' '.join(r)
        corpus.append(r)
    data['text'] = corpus

# Function to handle search
def search():
    if data is None:
        messagebox.showerror("Error", "Please load a CSV file first.")
        return

    query = search_entry.get()
    if not query:
        messagebox.showerror("Error", "Please enter a search query.")
        return

    tokenized_query = word_tokenize(query.lower())
    preprocessed_query = ' '.join(tokenized_query)

    # Create a TF-IDF vectorizer and calculate TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    query_vector = tfidf_vectorizer.transform([preprocessed_query])

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

    # Rank documents by similarity
    results = [(data.iloc[i], cosine_similarities[0][i]) for i in range(len(corpus))]
    results.sort(key=lambda x: x[1], reverse=True)

    # menghitung term tiap dokumen
    count_vectorizer = CountVectorizer()
    tf_matrix = count_vectorizer.fit_transform(corpus)

    # mengambil nama terms dengan menggunakan get_features_names_out
    terms = count_vectorizer.get_feature_names_out()
    tf_df = pd.DataFrame(tf_matrix.toarray(), columns=terms, index=list(data['text']))

    # menghitung TF pada query
    query_df = pd.DataFrame(0, columns=terms, index=[0])
    for term in tokenized_query:
        if term in query_df.columns:
            query_df.at[0, term] = 1

    # menghitung Weight
    weight = tf_matrix.multiply(query_df)
    weight_df = pd.DataFrame(weight.A, columns=terms, index=list(data['text']))

    # menghitung penyebut dengan mengkuadratkan term pada tiap dokumen |A^2|
    square_penyebut = tf_df.apply(np.square)

    # menghitung jumlah pembilang pada masing-masing document
    sum_pembilang = weight_df.sum(axis=1)

    # menghitung jumlah penyebut pada masing-masing document
    sum_penyebut = square_penyebut.sum(axis=1)

    # menghitung akar kuadrat dari jumlah penyebut masing-masing document
    sqrt_penyebut = np.sqrt(sum_penyebut)

    # melakukan operasi kuadrat pada term query |Q^2|
    square_query = query_df.apply(np.square)

    # menghitung akar kuadrat dari query |Q^2|
    sum_square_query = square_query.sum(axis=1)
    sqrt_square_query = np.sqrt(sum_square_query)

    # menghitung perkalian antara |A|.|B|
    multiply_documents = sqrt_penyebut * sqrt_square_query[0]

    # menghitung hasil similarities
    result_similarities = sum_pembilang / multiply_documents
    result_similarities_sorted = sorted(enumerate(result_similarities), key=lambda x: x[1], reverse=True)

    print(result_similarities)

    # mendisplaykan hasil
    num_doc_matched = 0
    search_result_frame.delete("1.0", tk.END)
    for i, similarity in result_similarities_sorted:
        if similarity > 0.00:
            num_doc_matched += 1
            row = data.iloc[i]
            result_text = f"{row['text']}\nPenulis: {row['penulis']}\nTahun: {row['tahun']}\n"
            search_result_frame.insert(tk.END, result_text)
            search_result_frame.insert(tk.END, "Document Link\n\n")
            search_result_frame.tag_add("link", "end-3c linestart", "end-2c lineend")
            search_result_frame.tag_config("link", foreground="blue", underline=1)
            search_result_frame.tag_bind("link", "<Button-1>", lambda e, url=row['link']: open_link(url))

    if num_doc_matched == 0:
        search_result_frame.insert(tk.END, "No item matched")

def event_search_btn(e):
    search()

def open_link(url):
    webbrowser.open_new(url)

load_file()

# Header
header_frame = tk.Frame(root)
header_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

header_label = tk.Label(header_frame, text="Information Retrieval System", font=("Arial", 16))
header_label.pack(side=tk.LEFT, padx=20)

# Search bar
search_frame = tk.Frame(root)
search_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

search_entry = tk.Entry(search_frame, font=("Arial", 14))
search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

search_button = tk.Button(search_frame, text="🔍", command=search, width=5)
search_button.pack(side=tk.RIGHT, padx=10)
# add event binding "enter key" to searching
root.bind('<Return>', event_search_btn)

# Main content
main_frame = tk.Frame(root)
main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)

# Search results section
search_result_frame = tk.Text(main_frame, bg="white")
search_result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Run the application
root.mainloop()