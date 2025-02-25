import pandas as pd
import os
import numpy as np
import spacy
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

nlp = spacy.load("en_core_web_sm")

folder_path = r'C:\devdev\Games\GOT'

def remove_stopwords(tokens):
    return [word for word in tokens if word not in STOPWORDS]

def process_chunk(text_chunk): 
    """Tokenize sentences, preprocess words, and remove stopwords."""
    doc = nlp(text_chunk)
    for sent in doc.sents:
        tokens = simple_preprocess(sent.text)  # Tokenize sentence
        filtered_tokens = remove_stopwords(tokens)  # Remove stopwords
        story.append(filtered_tokens)

story = []

for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)

    if file.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                chunk = f.read(500_000)  # Read 500,000 characters at a time
                if not chunk:
                    break
                process_chunk(chunk)  # Process each chunk separately

print(story[:5]) 


len(story)

from gensim.models import Word2Vec

model = Word2Vec(sentences=story, vector_size=100, window=5, min_count=3, workers=4)
model.save("word2vec_GOT.model")

model = Word2Vec.load("word2vec_GOT.model")  

for word, vocab_obj in model.wv.key_to_index.items():
    print(f"{word}: {model.wv.get_vecattr(word, 'count')} times")

model.wv.most_similar("raze")

import spacy
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = nlp.Defaults.stop_words
print(sorted(spacy_stopwords))

model = Word2Vec.load("word2vec_GOT.model")
nlp = spacy.load("en_core_web_sm")

CUSTOM_STOPWORDS = STOP_WORDS.union(nlp.Defaults.stop_words)

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in CUSTOM_STOPWORDS]

def get_paragraph_vector(paragraph, model):
    words = simple_preprocess(paragraph)  # Tokenize & preprocess
    clean_words = remove_stopwords(words)  # Remove stopwords
    word_vectors = [model.wv[word] for word in clean_words if word in model.wv]

    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average word embeddings
    else:
        return np.zeros(model.vector_size)  # Return zero vector if no words found

folder_path = r"GOT"

paragraph_vectors = []
paragraphs = []
file_names = []

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    if file_name.endswith('.txt'):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(500_000)  # Read in chunks
                if not chunk:
                    break
                
                doc = nlp(chunk)  # Tokenize sentences
                for para in doc.sents:
                    para_text = para.text.strip()
                    
                    words = simple_preprocess(para_text)
                    clean_words = remove_stopwords(words)
                    clean_para = " ".join(clean_words)

                    if len(clean_para) > 30:  # Ignore very short sentences
                        paragraphs.append(clean_para)  # Store cleaned paragraph
                        file_names.append(file_name)
                        paragraph_vectors.append(get_paragraph_vector(clean_para, model))

# Convert to numpy array for clustering
paragraph_vectors = np.array(paragraph_vectors)


from sklearn.cluster import KMeans

num_cluster=5

kmeans= KMeans(n_clusters=num_cluster, random_state=42)
cluster_labels= kmeans.fit_predict(paragraph_vectors)

# Organize paragraphs into clusters
clustered_paragraphs = {}
for cluster_id in range(num_cluster):
    clustered_paragraphs[cluster_id] = []

# Assign each paragraph to its cluster
for index, (file, para) in enumerate(zip(file_names, paragraphs)):
    cluster_id = cluster_labels[index]
    clustered_paragraphs[cluster_id].append((file, para))

for cluster_id, paras in clustered_paragraphs.items():
    print(f"\n🔹 **Cluster {cluster_id}** (Sample Paragraphs):")
    
    for file, para in paras[:3]:  
        print(f"📌 [{file}] {para[:100]}...\n")  


nlp = spacy.load("en_core_web_sm")

def get_top_words(cluster_paragraphs, top_n=10):
    word_counts = Counter()
    for _, para in cluster_paragraphs:
        word_counts.update(para.split())  
    return [word for word, _ in word_counts.most_common(top_n)]

def get_named_entities(cluster_paragraphs, top_n=5):
    """Extracts top named entities (characters, places, etc.) for cluster naming."""
    entity_counts = Counter()
    for _, para in cluster_paragraphs:
        doc = nlp(para)
        entity_counts.update([ent.text for ent in doc.ents])  
    return [entity for entity, _ in entity_counts.most_common(top_n)]

cluster_names = {}
for cluster_id, paras in clustered_paragraphs.items():
    top_words = get_top_words(paras)
    top_entities = get_named_entities(paras)

    topic_keywords = ", ".join(top_words[:3])
    entity_keywords = ", ".join(top_entities[:2])  

    if entity_keywords:
        cluster_names[cluster_id] = f"Topic: {entity_keywords} | Keywords: {topic_keywords}"
    else:
        cluster_names[cluster_id] = f"Topic: {topic_keywords}"

for cluster_id, name in cluster_names.items():
    print(f"Cluster {cluster_id} → {name}")
