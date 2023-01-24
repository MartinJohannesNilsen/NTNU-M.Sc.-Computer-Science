import re
import string
import gensim
from pathlib import Path
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

print("\n" + " -" * 20 + "\n  Assignment 3 - Martin Johannes Nilsen\n" + " -" * 20)

# Read book as string
book = Path("pg3300.txt").read_text("utf-8")

# Split into paragraphs
blank_line_regex = r"(?:\r?\n){2,}"
paragraphs = [str.rstrip("\n\r\t").replace("\n", " ")
              for str in re.split(blank_line_regex, book.strip())
              if str != "" and "gutenberg" not in str.lower()]
# The paragraph list now includes all paragraphs, without the paragraphs including the term "gutenberg", and with 5 lines of information at start and 14 paragraphs of disclosure

# Tokenize the paragraphs
paragraphs_tokenized = [str
                        # Lowercase
                        .lower()
                        # Strip away punctuation and new line characters
                        .translate(str.maketrans("", "", string.punctuation))
                        # Split on space
                        .split(" ")
                        for str in paragraphs]

# Stem the tokenized paragraphs
paragraphs_stemmed = [[stemmer.stem(word) for word in token_list]
                      for token_list in paragraphs_tokenized]

# Create dictionary
dictionary = gensim.corpora.Dictionary(paragraphs_stemmed)

# Filter out stopwords
stopwords = Path("stopwords.txt").read_text("utf-8").strip().split(",")
stopword_ids = [dictionary.token2id[stopword]
                for stopword in stopwords
                if stopword in dictionary.token2id]
dictionary.filter_tokens(stopword_ids)

# Create bag of words as a corpus, i.e. a bow list per paragraph
corpus = [dictionary.doc2bow(p) for p in paragraphs_stemmed]

# Build TF-IDF model, TF-IDF corpus and matrixSimilarity object
tfidf_model = gensim.models.TfidfModel(corpus)
tfidf_corpus = tfidf_model[corpus]
tfidf_matrix_similarity = gensim.similarities.MatrixSimilarity(tfidf_corpus)

# Same for LSI model
lsi_model = gensim.models.LsiModel(
    tfidf_corpus, id2word=dictionary, num_topics=100)
lsi_corpus = lsi_model[tfidf_corpus]
lsi_matrix_similarity = gensim.similarities.MatrixSimilarity(lsi_corpus)
# print(lsi_model.show_topics(num_topics=3))


# preprocess query
def preprocessing(query: str):
    query_tokenized = query.lower().translate(str.maketrans("", "", string.punctuation)).split(" ")
    query_stemmed = [stemmer.stem(word) for word in query_tokenized]
    return query_stemmed


query_unprocessed = "Query: 'What is the function of money?'"
# query_unprocessed = "Query: 'How taxes influence Economics?'"
print(query_unprocessed + "\n")
query = preprocessing(query_unprocessed)
query = dictionary.doc2bow(query)
tfidf_query = tfidf_model[query]

print("\nTask 4.2 - TFIDF weights of query\n" + "-" * 33, end="\n")
tfidf_weights = [f"{dictionary[token_id]}: {round(tfidf_weigth, 2)}" for token_id, tfidf_weigth in tfidf_query]
print(tfidf_weights)

# Find top 3 most relevant paragraphs for the query
doc2similarity = enumerate(tfidf_matrix_similarity[tfidf_query])
top_3_most_similar_paragraphs = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
# For truncating into 5 lines, I assume the perfect line length to be 75 characters, based on some quick research
print("\nTask 4.3 - Top 3 most relevant paragraphs\n" + "-" * 41)
for i, sim in top_3_most_similar_paragraphs:
    print(f"[paragraph {i+1}]")
    for line in [" ".join(paragraphs[i].split(" ")[x:x + 10]) for x in range(0, min(50, len(paragraphs[i].split(" "))), 10)]:
        print(line)
    print()

# Find the 3 most relevant topics
print("Task 4.4 - Top 3 topics and most relevant paragraphs\n" + "-" * 52, end="\n")
lsi_query = lsi_model[tfidf_query]
sorted_lsi = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3]
all_topics = lsi_model.show_topics()
for i, _ in sorted_lsi:
    print(f"[Topic {i}]\n{all_topics[i][1]}\n")

# Find the 3 most similar paragraphs using the lsi model
doc2similarity = enumerate(lsi_matrix_similarity[lsi_query])
top_3_most_similar_paragraphs = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
for i, sim in top_3_most_similar_paragraphs:
    # print(f"[paragraph {i+1}]\n{paragraphs[i][:75*5]}\n")
    print(f"\n[paragraph {i+1}]")
    for line in [" ".join(paragraphs[i].split(" ")[x:x + 10]) for x in range(0, min(50, len(paragraphs[i].split(" "))), 10)]:
        print(line)
