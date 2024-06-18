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

# Unduh data NLTK yang diperlukan
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Inisialisasi main window
root = tk.Tk()
root.title("Sistem Temu Kembali Informasi")
width = 800
height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = int((screen_width/2) - (width/2))
y_coordinate = int((screen_height/2) - (height/2) - 100)
# set jendela di tengah saat pertama kali dijalankan
root.geometry(f"{width}x{height}+{x_coordinate}+{y_coordinate}")
root.minsize(800, 600)

# Variabel global
data = None
corpus = []

def load_file():
    global data, corpus
    data = None
    corpus = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir = os.getcwd()      # Uncomment ini jika ingin dijalankan di Jupyter

    file = os.path.join(script_dir, "dataset.csv")
    
    try:
        data = pd.read_csv(file, encoding='latin-1')
        if len(data.columns) >= 2:
            data.columns = ['label', 'text', 'penulis', 'tahun', 'link']
        else:
            raise ValueError("File CSV harus memiliki minimal dua kolom")
        preprocess_text()
    except Exception as e:
        messagebox.showerror("Error", f"Error saat memuat file: {str(e)}")

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

    # Buat TF-IDF vectorizer dan hitung TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    query_vector = tfidf_vectorizer.transform([preprocessed_query])

    # Hitung term untuk setiap dokumen
    count_vectorizer = CountVectorizer()
    tf_matrix = count_vectorizer.fit_transform(corpus)

    # Dapatkan nama-nama term dengan menggunakan get_features_names_out
    terms = count_vectorizer.get_feature_names_out()
    tf_df = pd.DataFrame(tf_matrix.toarray(), columns=terms,
                         index=list(data['text']))

    # Hitung TF pada query
    query_tf = pd.DataFrame(0, columns=terms, index=[0])
    for term in tokenized_query:
        if term in query_tf.columns:
            query_tf.at[0, term] = 1

    # Hitung weight (W)
    weight = tf_matrix.multiply(query_tf)
    weight_df = pd.DataFrame(weight.A, columns=terms, index=list(data['text']))

    # Hitung penyebut dengan mengkuadratkan term pada tiap dokumen |A^2|
    square_penyebut = tf_df.apply(np.square)

    # Hitung jumlah pembilang pada masing-masing dokumen
    sum_pembilang = weight_df.sum(axis=1)

    # Hitung jumlah penyebut pada masing-masing dokumen
    sum_penyebut = square_penyebut.sum(axis=1)

    # Hitung akar kuadrat dari jumlah penyebut masing-masing dokumen
    sqrt_penyebut = np.sqrt(sum_penyebut)

    # Lakukan operasi kuadrat pada term query |Q^2|
    square_query = query_tf.apply(np.square)

    # Hitung akar kuadrat dari query |Q^2|
    sum_square_query = square_query.sum(axis=1)
    sqrt_square_query = np.sqrt(sum_square_query)

    # Hitung perkalian antara |A|.|B|
    multiply_documents = sqrt_penyebut * sqrt_square_query[0]

    # Hitung hasil similarities
    result_similarities = sum_pembilang / multiply_documents
    result_similarities_sorted = sorted(
        enumerate(result_similarities), key=lambda x: x[1], reverse=True)

    print(result_similarities)
    return result_similarities_sorted

# Fungsi untuk preprocessing teks
def preprocess_text():
    global corpus
    text = list(data['text'])
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    for i in range(len(text)):
        r = re.sub('[^a-zA-Z]', ' ', text[i])
        r = re.sub('[!@#$%^&*()_+-<>?/.:;"{}’[]|\]', ' ', text[i])
        r = r.lower()
        r = r.split()
        r = [word for word in r if word not in stopwords.words('indonesian')]
        r = [stemmer.stem(word) for word in r]
        r = ' '.join(r)
        corpus.append(r)
    data['text'] = corpus

# Fungsi untuk menangani pencarian
def search():
    if data is None:
        messagebox.showerror("Error", "Silakan muat file CSV terlebih dahulu.")
        return

    query = search_entry.get()
    if not query:
        messagebox.showerror("Error", "Silakan masukkan query pencarian.")
        return

    # panggil fungsi cosine_similarity_process
    result_similarities_sorted = cosine_similarity_process(query, corpus)
    
    # menampilkan hasil
    num_doc_matched = 0
    # hapus hasil sebelumnya
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

            # tambahkan label judul untuk menampilkan judul dokumen
            title_label = tb.Label(main_frame, text=row['text'].title(), foreground='blue', cursor='hand2',
                                   wraplength=1920, justify='left', anchor='w', font=underlined_font)
            title_label.pack(fill='x', pady=(0, 0), ipady=0)
            # tambahkan event bind untuk mengklik hyperlink judul dokumen
            title_label.bind("<Button-1>", lambda e,
                             url=row['link']: open_link(url))
            title_label.bind("<Enter>", label_hover_enter)
            title_label.bind("<Leave>", label_hover_leave)

            # tambahkan label penulis dan tahun publikasi dokumen
            author_label = tk.Label(main_frame, text=f"Penulis: {row['penulis']} | Tahun: {row['tahun']}", anchor='w')
            author_label.pack(fill='x')

            # tambahkan label similarity
            similarity_label = tk.Label(main_frame, text=f"Similarity: {similarity}", anchor='nw', font=(BOLD))
            similarity_label.pack(fill='x',  pady=(0, 20))
            # bisa tambahkan padding di sini

    total_item_founded.config(text=f"About {num_doc_matched} results")
    # untuk menambahkan scrollbar di main_frame
    main_frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

    # jika dokumen tidak ditemukan, tampilkan pesan
    if num_doc_matched == 0:
        no_result_label = tk.Label(
            main_frame, text="Did not match any documents", anchor='w')
        no_result_label.pack(fill='x', pady=(0, 12))
        canvas.configure(scrollregion=canvas.bbox("all"))

def label_hover_enter(event):
    event.widget.config(foreground='blue', font=('Arial', 16, 'underline'))

# Fungsi untuk event label hover leave
def label_hover_leave(event):
    event.widget.config(foreground='blue', font=('Arial', 16, 'normal'))

def event_search_btn(e):
    search()

def open_link(url):
    webbrowser.open_new(url)

# fungsi untuk wrap label judul
def update_wraplength(event=None):
    for widget in main_frame.winfo_children():
        widget.config(update_wraplength=main_frame.winfo_width())

# fungsi untuk scroll main_frame dengan mouse
def on_mouse_wheel_scroll(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

# fungsi untuk menghapus kata kunci
def clear_key():
    search_entry.delete(0, tk.END)
    for widget in main_frame.winfo_children():
        widget.destroy()
    # Reset scroll region
    canvas.configure(scrollregion=canvas.bbox("all"))

load_file()

# Header dan Search bar digabung dalam satu frame
header_search_frame = tk.Frame(root)
header_search_frame.pack(side=tk.TOP, fill=tk.X, pady=(30, 10))

header_label = tk.Label(header_search_frame, text="Scholarlink", font=("Arial", 24, "bold"))
header_label.pack(side=tk.LEFT, padx=20)

search_frame = tk.Frame(header_search_frame)
search_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

search_entry = tk.Entry(search_frame, font=("Arial", 18), width=40)
search_entry.pack(side=tk.LEFT, fill=tk.X, padx=(0, 0), ipady=5)

clear_keyword = tb.Button(search_frame, text='X', command=clear_key, bootstyle='primary')
clear_keyword.pack(side=tk.LEFT, ipadx=5, ipady=5)

search_button = tb.Button(search_frame, text="🔍", command=search, width=5, bootstyle='primary')
search_button.pack(side=tk.LEFT, fill=tk.X, padx=(10, 0), ipady=5)

# event binding "enter key" untuk pencarian
root.bind('<Return>', event_search_btn)

# tambahkan canvas
canvas = tk.Canvas(root)
canvas.pack(side='left', fill='both', expand=True, padx=20)

# tambahkan scrollbar untuk meng-scroll main_frame
scrollbar = tk.Scrollbar(root, orient='vertical', command=canvas.yview)
scrollbar.pack(side='right', fill='y')

# konfigurasi canvas
canvas.configure(yscrollcommand=scrollbar.set)

# Konten utama
main_frame = tk.Frame(canvas)

# tambahkan main_frame ke dalam canvas
canvas.create_window((0, 0), window=main_frame, anchor='nw')

def on_configure_scroll(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

# event bind untuk meng-scroll frame
canvas.bind("<Configure>", on_configure_scroll)
# event bind untuk meng-scroll frame dengan mousewheel
canvas.bind_all("<MouseWheel>", on_mouse_wheel_scroll)

# Jalankan aplikasi
root.mainloop()
