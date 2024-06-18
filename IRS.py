import os
import subprocess
pip_pkgs = ["nltk", "numpy", "pandas", "Sastrawi", "scikit-learn", "ttkbootstrap"]
for pip_pkg in pip_pkgs:
    subprocess.check_call(["pip", "install", pip_pkg])
        
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.font import BOLD, Font
import ttkbootstrap as tb
from ttkbootstrap.constants import *
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
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the main window
root = tk.Tk()
root.title("Information Retrieval System")
width = 1440
height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = int((screen_width/2) - (width/2))
y_coordinate = int((screen_height/2) - (height/2) - 100)
# set windows to center when it's run for first time
root.geometry(f"{width}x{height}+{x_coordinate}+{y_coordinate}")
root.minsize(800, 600)

# Global variables
data = None
corpus = []

def load_file():
    global data, corpus
    data = None
    corpus = []
    # script_dir = os.getcwd() # uncomment this if u wanna use it on jupyter or smth
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

def processed_query(query):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    query = re.sub('[^a-zA-Z]', ' ', query)
    query = re.sub('[!@#$%^&*()_+-<>?/.:;"{}’[]|\]', ' ', query)
    query = word_tokenize(query)
    query = [stemmer.stem(word.lower())
             for word in query if word not in stopwords.words('indonesian')]
    query = ' '.join(query)
    return query

def cosine_similarity_process(query, corpus):
    global data
    preprocessed_query = processed_query(query)
    tokenized_query = word_tokenize(preprocessed_query)

    # create a TF-IDF vectorixer and calculate TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    query_vector = tfidf_vectorizer.transform([preprocessed_query])

    # menghitung term tiap dokumen
    count_vectorizer = CountVectorizer()
    tf_matrix = count_vectorizer.fit_transform(corpus)

    # mengambil nama terms dengan menggunakan get_features_names_out
    terms = count_vectorizer.get_feature_names_out()
    tf_df = pd.DataFrame(tf_matrix.toarray(), columns=terms,
                         index=list(data['text']))

    # menghitung TF pada query
    query_tf = pd.DataFrame(0, columns=terms, index=[0])
    for term in tokenized_query:
        if term in query_tf.columns:
            query_tf.at[0, term] = 1

    # menghitung weight (W)
    weight = tf_matrix.multiply(query_tf)
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
    square_query = query_tf.apply(np.square)

    # menghitung akar kuadrat dari query |Q^2|
    sum_square_query = square_query.sum(axis=1)
    sqrt_square_query = np.sqrt(sum_square_query)

    # menghitung perkalian antara |A|.|B|
    multiply_documents = sqrt_penyebut * sqrt_square_query[0]

    # menghitung hasil similarities
    result_similarities = sum_pembilang / multiply_documents
    result_similarities_sorted = sorted(
        enumerate(result_similarities), key=lambda x: x[1], reverse=True)

    print(result_similarities)
    return result_similarities_sorted

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

    # call function cosine_similarity_process
    result_similarities_sorted = cosine_similarity_process(query, corpus)
    
    # mendisplaykan hasil
    num_doc_matched = 0
    # clear previous results
    for widget in main_frame.winfo_children():
        widget.destroy()
    total_item_founded = tk.Label(
        main_frame, text="", anchor='w', justify='left')
    total_item_founded.pack(fill='x', pady=(5, 16))
    
    for i, similarity in result_similarities_sorted:
        if similarity > 0.00:
            num_doc_matched += 1
            row = data.iloc[i]
            underlined_font = Font(
                family='Arial', size=16)

            # add label title to display the title document
            title_label = tb.Label(main_frame, text=row['text'].title(), foreground='blue', cursor='hand2',
                                   wraplength=1920, justify='left', anchor='w', font=underlined_font)
            title_label.pack(fill='x', pady=(0, 0), ipady=0)
            # add event bind to click the hyperlink title document
            title_label.bind("<Button-1>", lambda e,
                             url=row['link']: open_link(url))
            title_label.bind("<Enter>", label_hover_enter)
            title_label.bind("<Leave>", label_hover_leave)

            # add author and year of document publication
            author_label = tk.Label(main_frame, text=f"Penulis: {row['penulis']} | Tahun Terbit: {row['tahun']}", anchor='w')
            author_label.pack(fill='x', pady=(0, 20))
            # can you add padding here

    total_item_founded.config(text=f"Total item found: {num_doc_matched}")
    # for add scrollbar in main_frame
    main_frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

    # if document not found, display it
    if num_doc_matched == 0:
        no_result_label = tk.Label(
            main_frame, text="No Item Matched", anchor='w')
        no_result_label.pack(fill='x', pady=(0, 12))
        canvas.configure(scrollregion=canvas.bbox("all"))
def label_hover_enter(event):
    event.widget.config(foreground='blue', font=('Arial', 18, 'bold', 'underline'))
# Function to handle label hover leave event
def label_hover_leave(event):
    event.widget.config(foreground='blue', font=('Arial', 18, 'bold', 'normal'))
    
def event_search_btn(e):
    search()

def open_link(url):
    webbrowser.open_new(url)

# function to wrap label title
def update_wraplength(event=None):
    for widget in main_frame.winfo_children():
        widget.config(update_wraplength=main_frame.winfo_width())

# function to scroll the main_frame with mouse
def on_mouse_wheel_scroll(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

# function to clear keyword
def clear_key():
    search_entry.delete(0, tk.END)

load_file()

# Header
header_frame = tk.Frame(root)
header_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

header_label = tk.Label(header_frame, text="Scholarlink", font=("Arial", 24, "bold"))
header_label.pack(side=tk.LEFT, padx=20)

# Search bar
search_frame = tk.Frame(root)
search_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

search_entry = tk.Entry(search_frame, font=("Arial", 14), width=60)
search_entry.pack(side=tk.LEFT, fill=tk.X, padx=(20, 0), ipady=5)

clear_keyword = tb.Button(search_frame, text='x', command=clear_key, bootstyle='danger')
clear_keyword.pack(side=tk.LEFT, ipadx=5, ipady=5)

search_button = tb.Button(search_frame, text="🔍", command=search, width=5, bootstyle='primary')
search_button.pack(side=tk.LEFT, fill=tk.X, padx=(10, 0), ipady=5)


# add event binding "enter key" to searching
root.bind('<Return>', event_search_btn)

# add canvas
canvas = tk.Canvas(root)
canvas.pack(side='left', fill='both', expand=True, padx=20)

# add scrollbar to scroll the main_frame
scrollbar = tk.Scrollbar(root, orient='vertical', command=canvas.yview)
scrollbar.pack(side='right', fill='y')

# configure the canvas
canvas.configure(yscrollcommand=scrollbar.set)

# Main content
main_frame = tk.Frame(canvas)

# add main_frame to canvas
canvas.create_window((0, 0), window=main_frame, anchor='nw')

def on_configure_scroll(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

# add event bind to scroll frame
canvas.bind("<Configure>", on_configure_scroll)
# add event bind to scroll frame with mousewheel
canvas.bind_all("<MouseWheel>", on_mouse_wheel_scroll)

# Run the application
root.mainloop()
