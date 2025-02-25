# **Clustering Game of Thrones Words with Word2Vec & K-Means**  

## **📌 Project Overview**  
This project explores **Natural Language Processing (NLP)** by analyzing and clustering text from *Game of Thrones (GOT)* scripts/books. I experimented with **Word2Vec embeddings** to convert words into vector representations and applied **K-Means clustering** to group similar paragraphs.  

## **💡 What I Did**  
1. **Data Preprocessing**  
   - Loaded *Game of Thrones* text data.  
   - Tokenized and removed stopwords (including custom ones).  
   - Processed text into word embeddings using **Word2Vec**.  

2. **Feature Extraction**  
   - Generated paragraph vectors by averaging word embeddings.  

3. **Clustering**  
   - Applied **K-Means clustering** on the paragraph vectors.  
   - Assigned human-readable names to clusters based on the most frequent words.  

## **🎯 What I Learned**  
- How to use **Word2Vec** for feature extraction.  
- The importance of **stopword removal** (and debugging when they persist!).  
- How **K-Means** can be used for text clustering.  
- Practical challenges in **NLP tasks** like text cleaning, handling unseen words, and interpreting clusters.  

## **🚀 How to Run**  
1. Clone this repository:  
   ```sh
   git clone https://github.com/Vidushi2709/Clustering_of_GOT_words.git
   ```
2. Install dependencies:  
   ```sh
   pip install numpy gensim scikit-learn spacy
   python -m spacy download en_core_web_sm
   ```
3. Run the script:  
   ```sh
   python clustering_got.py
   ```

## **📌 Future Improvements**  
- Try **DBSCAN** or **Hierarchical Clustering** for better insights.  
- Fine-tune **Word2Vec** on *GOT-specific* vocabulary.  
- Visualize clusters using **t-SNE or PCA**.  

---
