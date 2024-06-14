import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import re
import nltk
import webbrowser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Function to handle CSV file selection
def select_file():
    global data, corpus
    data = None
    corpus = []
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    print(file_path)
    if file_path:
        try:
            data = pd.read_csv(file_path, encoding='latin-1')
            if len(data.columns) >= 2:
                data.columns = ['label', 'text', 'penulis', 'tahun', 'link']
            else:
                raise ValueError("CSV file must have at least two columns")
            preprocess_text()
            messagebox.showinfo("File Loaded", "CSV file loaded successfully.")
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

    # Display results
    search_result_frame.delete("1.0", tk.END)
    for row, similarity in results:
        if similarity > 0.00:
            result_text = f"{row['text']}\nPenulis: {row['penulis']}\nTahun: {row['tahun']}\n"
            search_result_frame.insert(tk.END, result_text)
            search_result_frame.insert(tk.END, "Document Link\n\n")
            search_result_frame.tag_add("link", "end-3c linestart", "end-2c lineend")
            search_result_frame.tag_config("link", foreground="blue", underline=1)
            search_result_frame.tag_bind("link", "<Button-1>", lambda e, url=row['link']: open_link(url))

def open_link(url):
    webbrowser.open_new(url)

# Header
header_frame = tk.Frame(root)
header_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

header_label = tk.Label(header_frame, text="Information Retrieval System", font=("Arial", 16))
header_label.pack(side=tk.LEFT, padx=20)

select_file_button = tk.Button(header_frame, text="Select CSV file", command=select_file)
select_file_button.pack(side=tk.RIGHT, padx=20)

# Search bar
search_frame = tk.Frame(root)
search_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

search_entry = tk.Entry(search_frame, font=("Arial", 14))
search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

search_button = tk.Button(search_frame, text="🔍", command=search)
search_button.pack(side=tk.RIGHT, padx=20)

# Main content
main_frame = tk.Frame(root)
main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)

# Search results section
search_result_frame = tk.Text(main_frame, bg="white")
search_result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Run the application
root.mainloop()
