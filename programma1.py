# Programma 1 di Andrea Belliani, 596864, Progetto LC a.a. 2024-2025

# Importo le librerie che mi servono
import nltk
import string
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Download risorse necessarie di NLTK
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('punkt')

# funzione che legge in input il file e restituisce il contenuto
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as infile:
            text = infile.read()
        return text
    except FileNotFoundError:
        print("Errore: file non trovato")
        return None
    except Exception as e:
        print(f"Errore imprevisto: {e}")
        return None

# funzione che segmenta il testo in frasi e ne calcola il numero
def get_sentences_and_length(text):
    try:
        sentences = nltk.sent_tokenize(text) # Segmento il testo in frasi
        sentences_length = len(sentences) # Calcolo il numero di frasi
        return sentences, sentences_length
    except Exception as e:
        print(f"Errore imprevisto: {e}")
        return None

# funzione che segmenta il testo in token e ne calcola il numero
def get_tokens_and_length(sentences):
    try:
        all_tokens = []
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)  # Tokenizzo ogni frase
            all_tokens.extend(tokens)  # Aggiungo i token alla lista totale
        length_corpus = len(all_tokens)
        return all_tokens, length_corpus
    except Exception as e:
        print(f"Errore durante la tokenizzazione: {e}")
        return None, 0

# funzione che calcola la media della lunghezza delle frasi in token
def get_avg_sentences_length(sentences_length, length_corpus):
    try:
        avg_sentences_length = length_corpus / sentences_length  # Calcolo la media della lunghezza delle frasi in token
        return avg_sentences_length
    except Exception as e:
        print(f"Errore durante il calcolo della media: {e}")
        return 0
       
# funzione che che calcola la media dei token (punteggiatura esclusa)
def get_avg_tokens_length(all_tokens):
    try:
        punctuation = set(string.punctuation) # Creo un set con i caratteri di punteggiatura
        tokens_without_punct = [token for token in all_tokens if token not in punctuation] # Creo una lista di token senza punteggiatura
        tokens_without_punct_length = len(tokens_without_punct) # Calcolo la lunghezza della lista di token senza punteggiatura
        sum_tokens_length = 0
        for token in tokens_without_punct:
            sum_tokens_length += len(token) # Calcolo la somma delle lunghezze dei token senza punteggiatura
        avg_tokens_length = sum_tokens_length / tokens_without_punct_length if tokens_without_punct_length > 0 else 0 # Calcolo la media delle lunghezze dei token
        return avg_tokens_length
    except Exception as e:
        print(f"Errore durante il calcolo della media: {e}")
        return 0

# funzione che calcola la distribuzione delle parti del discorso
def get_pos_distribution(all_tokens):
    try:
        pos_tags = nltk.pos_tag(all_tokens) # Calcolo le parti del discorso dei token
        pos_only = [pos for token, pos in pos_tags] # Creo una lista con le sole parti del discorso
        pos_distribution = FreqDist(pos_only) # Calcolo la distribuzione delle parti del discorso
        return dict(pos_distribution)
    except Exception as e:
        print(f"Errore durante il calcolo delle parti del discorso: {e}")
        return None

# funzione che calcola la dimensione del vocabolario e la ttr
def get_vocabulary_and_ttr(all_tokens):
    try:
        TTR_V_SLICING = 200  # Dimensione della finestra incrementale
        length_corpus = len(all_tokens)  # Lunghezza totale del corpus
        all_length_vocabulary = []  # Lista per salvare le dimensioni del vocabolario
        all_ttr = []  # Lista per salvare i TTR calcolati

        step = 200  # Inizio con i primi 200 token
        while step <= length_corpus:  
            if step + TTR_V_SLICING > length_corpus:
                step = length_corpus  # Limita 'step' alla lunghezza totale del corpus

            partial_tokens = all_tokens[:step]  # Prendo i primi 'step' token

            # Calcolo il vocabolario e il TTR per questa finestra
            vocabulary = list(set(partial_tokens))  # Elenco di parole uniche (vocabolario)
            length_vocabulary = len(vocabulary)  # Numero di parole uniche (dimensione del vocabolario)
            ttr = length_vocabulary / len(partial_tokens)  # Calcolo del TTR

            # Aggiungo i risultati alle liste
            all_length_vocabulary.append(length_vocabulary)
            all_ttr.append(ttr)

            # Incremento
            step += TTR_V_SLICING
        # Restituisco i risultati finali
        return all_ttr, all_length_vocabulary

    except ZeroDivisionError:
        print("Errore: Il corpus è vuoto, impossibile calcolare il TTR.")
        return None
    except Exception as e:
        print(f"Errore imprevisto: {e}")
        return None

# funzione per la lemmatizzazione dei token
def get_lemmas_and_length(all_tokens):
    try:
        lemmatizer = WordNetLemmatizer()  # Inizializzo il lemmatizzatore
        lemmas = [lemmatizer.lemmatize(token) for token in all_tokens]  # Lemmatizzo i token
        length_lemmas = len(set(lemmas))  # Calcolo il numero di lemmi unici
        return length_lemmas
    except Exception as e:
        print(f"Errore durante la lemmatizzazione: {e}")
        return None, 0

# funzione che calcola il numero medio di lemmi per frase
def get_avg_lemmas_per_sentence(sentences_length, length_lemmas):
    try:
        avg_lemmas_per_sentence = length_lemmas / sentences_length # Calcolo il numero medio di lemmi per frase
        return avg_lemmas_per_sentence
    except Exception as e:
        print(f"Errore durante il calcolo della media: {e}")
        return 0

# funzione che analizza la polarità delle frasi
def polarity_analyze(sentences, dataset_path="movie_reviews"):
    try:
        # Caricamento del dataset
        films_dataset = load_files(dataset_path, shuffle=True)

        # Divisione in training set e test set
        X_train, X_test, y_train, y_test = train_test_split(
            films_dataset.data, films_dataset.target, test_size=0.20, random_state=32
        )

        # Creazione della pipeline
        vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, max_features=3000)
        classifier = MultinomialNB()
        pipeline = Pipeline([
            ('tfidf-vectorizer', vectorizer),
            ('multinomialNB', classifier)
        ])

        # Addestramento del modello
        trained_model = pipeline.fit(X_train, y_train)

        # Test del modello
        predicted = trained_model.predict(X_test)

        # Calcolo l'accuratezza del modello
        classification_report(y_test, predicted, target_names=films_dataset.target_names)

        # Predizione
        prediction = list(trained_model.predict(sentences))
        return prediction
    except FileNotFoundError:
        print(f"Errore: Il dataset non è stato trovato nel percorso specificato '{dataset_path}'.")
    except Exception as e:
        print(f"Errore durante l'analisi della polarità: {e}")
        return None

# funzione che calcola distribuzione delle frasi positive e negative
def polarity_distribution(prediction):
    try:
        negative = prediction.count(0)
        positive = prediction.count(1)
        negative_distribution = negative / len(prediction) * 100 if len(prediction) > 0 else 0 # Calcolo la distribuzione delle frasi negative
        positive_distribution = positive / len(prediction) * 100 if len(prediction) > 0 else 0 # Calcolo la distribuzione delle frasi positive
        return positive_distribution, negative_distribution
    except Exception as e:
        print(f"Errore durante il calcolo della distribuzione: {e}")
        return 0, 0

#funzione che calcola la polarità del corpus
def total_polarity(prediction):
    try:
        corpus_polarity = sum(prediction) # Calcolo la polarità totale del corpus
        return corpus_polarity
    except Exception as e:
        print(f"Errore durante il calcolo della polarità: {e}")
        return 0

# funzione principale che analizza i due corpora richiamando le funzioni ausiliarie e salva i risultati in un file di output
def main(file_path1, file_path2):
    OUTPUT_FILE = "output1.txt"
    POS_SLICING = 1000

    text1 = read_file(file_path1)
    text2 = read_file(file_path2)
    
    sentences1, sentences_length1 = get_sentences_and_length(text1)
    sentences2, sentences_length2 = get_sentences_and_length(text2)
        
    tokens1, length_corpus1 = get_tokens_and_length(sentences1)
    tokens2, length_corpus2 = get_tokens_and_length(sentences2)
        
    avg_sentences_length1 = get_avg_sentences_length(sentences_length1, length_corpus1)
    avg_sentences_length2 = get_avg_sentences_length(sentences_length2, length_corpus2)
        
    avg_tokens_length1 = get_avg_tokens_length(tokens1)
    avg_tokens_length2 = get_avg_tokens_length(tokens2)
        
    partial_tokens1 = [token for token in tokens1[:(POS_SLICING)]]
    partial_tokens2 = [token for token in tokens2[:(POS_SLICING)]]
    pos_distribution1 = get_pos_distribution(partial_tokens1)
    pos_distribution2 = get_pos_distribution(partial_tokens2)
    all_pos = list(set(pos_distribution1.keys()).union(set(pos_distribution2.keys()))) # Unione delle chiavi dei due dizionari per confrontare le parti del discorso nei due corpora
        
    ttr1, length_vocabulary1 = get_vocabulary_and_ttr(tokens1)
    ttr2, length_vocabulary2 = get_vocabulary_and_ttr(tokens2)

    length_lemmas1 = get_lemmas_and_length(tokens1)
    length_lemmas2 = get_lemmas_and_length(tokens2)

    avg_lemmas_per_sentence1 = get_avg_lemmas_per_sentence(sentences_length1, length_lemmas1)
    avg_lemmas_per_sentence2 = get_avg_lemmas_per_sentence(sentences_length2, length_lemmas2)

    polarity1 = polarity_analyze(sentences1)
    polarity2 = polarity_analyze(sentences2)

    positive_sentences1, negative_sentences1 = polarity_distribution(polarity1)
    positive_sentences2, negative_sentences2 = polarity_distribution(polarity2)

    total_polarity1 = total_polarity(polarity1)
    total_polarity2 = total_polarity(polarity2)

    with open(OUTPUT_FILE, 'w') as outfile: # Salva i risultati nel file di output
        outfile.write(f"Risultati Corpus 1 - The Picture of Dorian Gray (Chapter I):\nNumero di frasi: {sentences_length1}\nNumero di token: {length_corpus1}\nMedia lunghezza frasi: {avg_sentences_length1}\nMedia lunghezza token: {avg_tokens_length1}\nDistribuzione PoS primi 1000 token: {pos_distribution1}\nDimensione del vocabolario calcolata ogni 200 token: {length_vocabulary1}\nTTR calcolata ogni 200 token: {ttr1}\nNumeno di lemmi distinti: {length_lemmas1}\nNumero medio di lemmi per frase: {avg_lemmas_per_sentence1}\nDistribuzione di frasi positive: {positive_sentences1}\nDistribuzione di frasi negative: {negative_sentences1}\nPolarità del documento: {total_polarity1}\n\n")
        outfile.write(f"Risultati Corpus 2 - The Exacting, Expansive Mind of Christopher Nolan:\nNumero di frasi: {sentences_length2}\nNumero di token: {length_corpus2}\nMedia lunghezza frasi: {avg_sentences_length2}\nMedia lunghezza token: {avg_tokens_length2}\nDistribuzione PoS primi 1000 token: {pos_distribution2}\nDimensione del vocabolario calcolata ogni 200 token: {length_vocabulary2}\nTTR calcolata ogni 200 token: {ttr2}\nNumeno di lemmi distinti: {length_lemmas2}\nNumero medio di lemmi per frase: {avg_lemmas_per_sentence2}\nDistribuzione di frasi positive: {positive_sentences2}\nDistribuzione di frasi negative: {negative_sentences2}\nPolarità del documento: {total_polarity2}\n\n")
        outfile.write("Distribuzione delle parti del discorso (PoS):\n")
        outfile.write("{:<10}  {:<10}  {:<10}\n".format("PoS", "Corpus 1", "Corpus 2"))  # Aggiungo formattazione per le intestazioni delle colonne
        for pos in all_pos:
            outfile.write("{:<10}  {:<10}  {:<10}\n".format(pos, pos_distribution1.get(pos, 0), pos_distribution2.get(pos, 0))) # Aggiungo formattazione per i dati delle colonne

if __name__ == "__main__":
    # Definisco i percorsi dei file
    INPUT_FILE1 = "corpus1.txt"
    INPUT_FILE2 = "corpus2.txt"

    # Eseguo la funzione principale
    main(INPUT_FILE1, INPUT_FILE2)