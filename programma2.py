# Programma 2 di Andrea Belliani, 596864, Progetto LC a.a. 2024-2025

import nltk
import math
from nltk import FreqDist
from collections import Counter
from nltk.corpus import stopwords 

# Download risorse necessarie di NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Funzione che legge in input il file e restituisce il contenuto
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

# Funzione che segmenta il testo in frasi e ne calcola il numero
def get_sentences_and_length(text):
    try:
        sentences = nltk.sent_tokenize(text)  # Segmenta il testo in frasi
        sentences_length = len(sentences)  # Calcola il numero di frasi
        return sentences, sentences_length
    except Exception as e:
        print(f"Errore imprevisto: {e}")
        return None, 0

# Funzione che segmenta il testo in token e ne calcola il numero
def get_tokens_and_length(sentences):
    try:
        all_tokens = []
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)  # Tokenizza ogni frase
            all_tokens.extend(tokens)  # Aggiunge i token alla lista totale
        length_corpus = len(all_tokens)
        return all_tokens, length_corpus
    except Exception as e:
        print(f"Errore durante la tokenizzazione: {e}")
        return None, 0

# Funzione che calcola le parti del discorso
def get_pos_tag(all_tokens):
    try:
        pos_tags = nltk.pos_tag(all_tokens)  # Calcola le parti del discorso dei token
        pos_only = [pos for token, pos in pos_tags]  # Crea una lista con le sole parti del discorso
        return pos_tags, set(pos_only)
    except Exception as e:
        print(f"Errore durante il calcolo delle parti del discorso: {e}")
        return None, {}

# funzione che trova i top-50 Sostantivi, Avverbi e Aggettivi più frequenti
def get_top_pos(pos_tags):
    try:
        TOP_N = 50
        top_nouns = FreqDist([token for token, pos in pos_tags if pos.startswith('N')]).most_common(TOP_N)  # Trova i top-50 sostantivi
        top_adverbs = FreqDist([token for token, pos in pos_tags if pos.startswith('R')]).most_common(TOP_N)  # Trova i top-50 avverbi
        top_adjectives = FreqDist([token for token, pos in pos_tags if pos.startswith('J')]).most_common(TOP_N)  # Trova i top-50 aggettivi
        return dict(top_nouns), dict(top_adverbs), dict(top_adjectives)
    except Exception as e:
        print(f"Errore durante il calcolo dei top-50 POS: {e}")
        return None, None, None
    
# funzione che trova i top-20 n-grammi più frequenti, con relativa frequenza e ordinati per frequenza decrescente per n = [1,2,3]
def get_top_ngrams(all_tokens):
    try:
        TOP_N = 20
        top_ngrams = []
        for n in range(1, 4):
            ngrams = list(nltk.ngrams(all_tokens, n))
            ngrams_freq = FreqDist(ngrams)
            common_ngrams = ngrams_freq.most_common(TOP_N)
            for i in range(TOP_N):
                top_ngrams.append((common_ngrams[i][0], common_ngrams[i][1]))
        return top_ngrams
    except Exception as e:
        print(f"Errore durante il calcolo dei top-20 n-grammi: {e}")
        return None
    
# funzione che trova i top 20 n-grammi di PoS più frequenti con relativa frequenza, e ordinati per frequenza decrescente per n = [1, 2, 3, 4, 5]
def get_top_pos_ngrams(pos_only):
    try:
        TOP_N = 20
        top_pos_ngrams = []
        for n in range(1, 6):
            ngrams = list(nltk.ngrams(pos_only, n))
            ngrams_freq = FreqDist(ngrams)
            common_ngrams = ngrams_freq.most_common(TOP_N)
            for i in range(TOP_N):
                top_pos_ngrams.append((common_ngrams[i][0], common_ngrams[i][1]))
        return top_pos_ngrams
    except Exception as e:
        print(f"Errore durante il calcolo dei top-20 n-grammi di PoS: {e}")
        return None
    
# funzione che trova i top-10 bigrammi composti da Verbo e Sostantivo, ordinati per: 
#a. frequenza decrescente, con relativa frequenza
def get_top_bigrams(pos_tags):
    try:
        TOP_N = 10
        bigrams = list(nltk.bigrams(pos_tags))  # Trova i bigrammi
        bigrams_freq = FreqDist(bigrams)  # Calcola la frequenza dei bigrammi
        # Filtro i bigrammi che iniziano con un verbo e finiscono con un nome
        filtered_bigrams = FreqDist()
        for bigram, freq in bigrams_freq.items():
            if bigram[0][1].startswith('V') and bigram[1][1].startswith('N'):  # Verbo -> Nome
                filtered_bigrams[(bigram[0][0], bigram[1][0])] = freq
        top_bigrams = filtered_bigrams.most_common(TOP_N)
        return dict(filtered_bigrams), top_bigrams
    except Exception as e:
        print(f"Errore durante il calcolo dei top-10 bigrammi: {e}")
        return None, None
    
#b. probabilità condizionata massima, e relativo valore di probabilità
def get_max_conditional_probability(filtered_bigrams, all_tokens):
    try:
        TOP_N = 10
        conditional_probabilities = {}
        token_frequencies = Counter(all_tokens) # Frequenze dei token
        for bigram, freq in filtered_bigrams.items():
            token1 = bigram[0]
            token1_freq = token_frequencies.get(token1, 0)
            conditional_probability = freq / token1_freq  # Probabilità condizionata 
            conditional_probabilities[(bigram[0], bigram[1])] = conditional_probability

        # Ordina i bigrammi per probabilità condizionata decrescente e restituisce i top N
        sorted_conditional_probability = sorted(conditional_probabilities.items(), key=lambda x: x[1], reverse=True)
        max_conditional_probability = sorted_conditional_probability[:TOP_N]  # Ottieni i top N
        return max_conditional_probability
    except Exception as e:  
        print(f"Errore durante il calcolo della probabilità condizionata massima: {e}")
        return None
    
#c. probabilità congiunta massima, e relativo valore di probabilità 
def get_max_joint_probability(max_conditional_probability, all_tokens, length_corpus):
    try:
        TOP_N = 10
        joint_probabilities = {}
        token_frequencies = Counter(all_tokens) # Frequenze dei token
        for bigram, cond_prob in max_conditional_probability:
            token1 = bigram[0]
            token1_freq = token_frequencies.get(token1, 0)
            joint_probability = cond_prob * token1_freq / length_corpus  # Probabilità congiunta
            joint_probabilities[(bigram[0], bigram[1])] = joint_probability

        # Ordina i bigrammi per probabilità congiunta decrescente e restituisce i top N
        sorted_joint_probability = sorted(joint_probabilities.items(), key=lambda x: x[1], reverse=True)
        max_joint_probability = sorted_joint_probability[:TOP_N]  # Ottieni i top N
        return max_joint_probability
    except Exception as e:
        print(f"Errore durante il calcolo della probabilità congiunta massima: {e}")
        return None
    
#d-e. MI (Mutual Information) e LMI (Local Mutual Information) massima, e relativo valore di MI e LMI
def get_max_MI_and_LMI(filtered_bigrams, all_tokens, length_corpus):
    try:
        TOP_N = 10
        mutual_informations = {}
        local_mutual_informations = {}
        token_frequencies = Counter(all_tokens) # Frequenze dei token
        for bigram, freq in filtered_bigrams.items():
            token1 = bigram[0]
            token2 = bigram[1]
            token1_freq = token_frequencies.get(token1, 0)
            token2_freq = token_frequencies.get(token2, 0)
            bigram_freq = freq
            mutual_information = math.log((bigram_freq * length_corpus / (token1_freq * token2_freq)), 2)  # Mutual Information
            local_mutual_information = mutual_information * bigram_freq  # Local Mutual Information
            mutual_informations[(bigram[0], bigram[1])] = mutual_information
            local_mutual_informations[(bigram[0], bigram[1])] = local_mutual_information

        # Ordina i bigrammi per Mutual Information decrescente e restituisce i top N
        sorted_mutual_information = sorted(mutual_informations.items(), key=lambda x: x[1], reverse=True)
        sorted_local_mutual_information = sorted(local_mutual_informations.items(), key=lambda x: x[1], reverse=True)
        max_mutual_information = sorted_mutual_information[:TOP_N]  # Ottieni i top N
        max_local_mutual_information = sorted_local_mutual_information[:TOP_N]  # Ottieni i top N
        return max_mutual_information, max_local_mutual_information
    except Exception as e:
        print(f"Errore durante il calcolo della Mutual Information massima: {e}")
        return None
    
#f. Calcolare e stampare il numero di elementi comuni ai top-10 per MI e per LMI
def get_common_elements(max_mutual_information, max_local_mutual_information):
    try:
        common_elements = set(max_mutual_information).intersection(set(max_local_mutual_information))
        return common_elements, len(common_elements)
    except Exception as e:
        print(f"Errore durante il calcolo degli elementi comuni: {e}")
        return 0 

# Considerare le frasi con una lunghezza compresa tra 10 e 20 token, in cui almeno la metà (considerare la parte intera della divisione per due come valore) dei token occorre almeno 2 volte nel corpus (i.e., non è un hapax)
def get_sentences_with_length(sentences):
    try:
        filtered_sentences = []
        non_hapax = []
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            if 10 <= len(tokens) <= 20:
                token_frequencies = FreqDist(tokens)
                for token, freq in token_frequencies.items():
                    if freq >= 2:
                        non_hapax.append(token)
                if len(non_hapax) >= len(tokens) // 2:
                    filtered_sentences.append(sentence)
        return filtered_sentences
    except Exception as e:
        print(f"Errore durante la selezione delle frasi: {e}")
        return None
    
# funzione che trova la frase con la media della distribuzione di frequenza dei token più alta e più bassa
def get_sentences_frequency(filtered_sentences):
    try:
        freq_sentences = {}
        for sentence in filtered_sentences:
            tokens = nltk.word_tokenize(sentence)
            token_frequencies = FreqDist(tokens)
            avg_freq = sum(token_frequencies.values()) / len(token_frequencies)
            freq_sentences[sentence] = avg_freq
        max_sentence = max(freq_sentences, key=freq_sentences.get) # Trova la frase con media di frequenza più alta
        min_sentence = min(freq_sentences, key=freq_sentences.get) # Trova la frase con media di frequenza più bassa
        max_avg_freq = freq_sentences[max_sentence] # Calcola la media di frequenza più alta
        min_avg_freq = freq_sentences[min_sentence] # Calcola la media di frequenza più bassa
        return max_sentence, min_sentence, max_avg_freq, min_avg_freq
    except Exception as e:
        print(f"Errore durante il calcolo delle frasi con media di frequenza più alta e più bassa: {e}")
        return None, None, 0, 0
    
# funzione che trova la frase con probabilità più alta secondo un modello di Markov di ordine 2 costruito a partire dal corpus di input
def get_markov_order_2(all_tokens):
    try:
        # Trova i bigrammi e trigrammi
        bigrams = list(nltk.bigrams(all_tokens))
        trigrams = list(nltk.trigrams(all_tokens))
        
        # Calcola le frequenze
        bigram_freq = FreqDist(bigrams)
        trigram_freq = FreqDist(trigrams)
        token_frequencies = FreqDist(all_tokens)
        
        # Calcola le probabilità condizionate dei bigrammi
        bigrams_conditional_probability = {}
        for bigram, freq in bigram_freq.items():
            token1 = bigram[0]
            token1_freq = token_frequencies.get(token1, 0)
            conditional_probability = freq / token1_freq  # P(token2 | token1)
            bigrams_conditional_probability[bigram] = conditional_probability
        
        # Calcola le probabilità condizionate dei trigrammi
        trigrams_conditional_probability = {}
        for trigram, freq in trigram_freq.items():
            token1_token2 = (trigram[0], trigram[1])
            token1_token2_freq = bigram_freq.get(token1_token2, 0)
            conditional_probability = freq / token1_token2_freq  # P(token3 | token1, token2)
            trigrams_conditional_probability[trigram] = conditional_probability
        return bigrams_conditional_probability, trigrams_conditional_probability
    except Exception as e:
        print(f"Errore durante il calcolo delle probabilità condizionate: {e}")
        return None, None

def get_max_markov_probability(filtered_sentences, bigrams_conditional_probability, trigrams_conditional_probability):
    try:
        max_markov_probability = 0.0
        max_markov_sentence = ""
        for sentence in filtered_sentences:
            markov_probability = 1.0
            tokens = nltk.word_tokenize(sentence)
            bigrams = list(nltk.bigrams(tokens))
            trigrams = list(nltk.trigrams(tokens))
            for bigram in bigrams:
                if bigram in bigrams_conditional_probability:
                    markov_probability *= bigrams_conditional_probability[bigram]
            for trigram in trigrams:
                if trigram in trigrams_conditional_probability:
                    markov_probability *= trigrams_conditional_probability[trigram]
            if markov_probability > max_markov_probability:
                max_markov_probability = markov_probability
                max_markov_sentence = sentence
        return max_markov_sentence, max_markov_probability
    except Exception as e:
        print(f"Errore durante il calcolo della probabilità di Markov: {e}")
        return None, 0
    
# funzione che trova la percentuale di Stopwords nel corpus rispetto al totale dei token
def get_stopwords_percentage(all_tokens, length_corpus):
    try:
        stop_words = set(stopwords.words('english'))
        stopwords_count = 0
        for token in all_tokens:
            if token in stop_words:
                stopwords_count += 1
        stopwords_percentage = (stopwords_count / length_corpus) * 100
        return stopwords_percentage
    except Exception as e:
        print(f"Errore durante il calcolo della percentuale di stopwords: {e}")
        return 0

# funzione che trova numero di pronomi personali sul totale di token e numero medio di pronomi personali per frase
def get_personal_pronouns(pos_tags, sentences_length):
    try:
        personal_pronouns = []
        for token, pos in pos_tags:
            if pos == 'PRP':
                personal_pronouns.append(token)
        personal_pronouns_count = len(personal_pronouns)
        avg_personal_pronouns = personal_pronouns_count / sentences_length
        return personal_pronouns_count, avg_personal_pronouns
    except Exception as e:
        print(f"Errore durante il calcolo dei pronomi personali: {e}")
        return 0, 0
    
# funzione che estratte le Entità Nominate del testo, identifica per ciascuna classe di NE i 15 elementi più frequenti, ordinati per frequenza decrescente e con relativa frequenza
def get_named_entities(all_tokens):
    try:
        TOP_NER = 15
        named_entities = nltk.ne_chunk(nltk.pos_tag(all_tokens))  # Estrae le Named Entities
        named_entities_freq = FreqDist()
        for entity in named_entities:
            if isinstance(entity, nltk.Tree):
                entity_label = entity.label()
                entity_words = ' '.join([token for token, pos in entity.leaves()])
                named_entities_freq[(entity_label, entity_words)] += 1

        top_named_entities = named_entities_freq.most_common(TOP_NER)
        return dict(top_named_entities)
    except Exception as e:
        print(f"Errore durante l'estrazione delle Named Entities: {e}")
        return None

# funzione principale che chiama tutte le funzioni ausiliarie e stampa i risultati in output
def main(file_path):
    OUTPUT_FILE = f"output2_{file_path}"

    text = read_file(file_path)

    sentences, sentences_length = get_sentences_and_length(text)
    all_tokens, length_corpus = get_tokens_and_length(sentences)
    pos_tags, pos_only = get_pos_tag(all_tokens)

    top_nouns, top_adverbs, top_adjectives = get_top_pos(pos_tags)
    top_ngrams = get_top_ngrams(all_tokens)
    top_pos_ngrams = get_top_pos_ngrams(pos_only)

    filtered_bigrams, top_bigrams = get_top_bigrams(pos_tags)
    max_conditional_probability = get_max_conditional_probability(filtered_bigrams, all_tokens)
    max_joint_probability = get_max_joint_probability(max_conditional_probability, all_tokens, length_corpus)
    max_mutual_information, max_local_mutual_information = get_max_MI_and_LMI(filtered_bigrams, all_tokens, length_corpus)
    common_elements, common_elements_count = get_common_elements(max_mutual_information, max_local_mutual_information)


    filtered_sentences = get_sentences_with_length(sentences)
    max_sentence, min_sentence, max_avg_freq, min_avg_freq = get_sentences_frequency(filtered_sentences)
    bigrams_conditional_probability, trigrams_conditional_probability = get_markov_order_2(all_tokens)
    max_markov_sentence, max_markov_probability = get_max_markov_probability(filtered_sentences, bigrams_conditional_probability, trigrams_conditional_probability)

    stopwords_percentage = get_stopwords_percentage(all_tokens, length_corpus)

    personal_pronouns_count, avg_personal_pronouns = get_personal_pronouns(pos_tags, sentences_length)

    top_named_entities = get_named_entities(all_tokens)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        if file_path == "corpus1.txt":
            outfile.write("Risultati Corpus 1 - The Picture of Dorian Gray (Chapter I):\n")
        else:
            outfile.write("Risultati Corpus 2 - The Exacting, Expansive Mind of Christopher Nolan:\n")
        outfile.write(f"Top-50 Sostantivi: \n{top_nouns}\n")
        outfile.write(f"\nTop-50 Avverbi: \n{top_adverbs}\n")
        outfile.write(f"\nTop-50 Aggettivi: \n{top_adjectives}\n")
        for n in range(1, 4):
            outfile.write(f"\nTop-20 n-grammi per n={n}: \n{top_ngrams[(n-1)*20:n*20]}\n")
        for n in range(1, 6):
            outfile.write(f"\nTop-20 n-grammi di PoS per n={n}: \n{top_pos_ngrams[(n-1)*20:n*20]}\n")
        outfile.write(f"\nTop-10 Bigrammi Verbo-Sostantivo ordinati per frequenza decrescente: \n{top_bigrams}\n")
        outfile.write(f"\nTop-10 Bigrammi Verbo-Sostantivo ordinati per probabilità condizionata massima: \n{max_conditional_probability}\n")
        outfile.write(f"\nTop-10 Bigrammi Verbo-Sostantivo ordinati per probabilità congiunta massima: \n{max_joint_probability}\n")
        outfile.write(f"\nTop-10 Bigrammi Verbo-Sostantivo ordinati per MI massima: \n{max_mutual_information}\n")
        outfile.write(f"\nTop-10 Bigrammi Verbo-Sostantivo ordinati per LMI massima: \n{max_local_mutual_information}\n")
        outfile.write(f"\nElementi comuni e numero di elementi comuni ai Top-10 per MI e LMI: \n{common_elements} ({common_elements_count})\n")
        outfile.write(f"\nFrase con media di frequenza più alta: \n{max_sentence} ({max_avg_freq})\n")
        outfile.write(f"\nFrase con media di frequenza più bassa: \n{min_sentence} ({min_avg_freq})\n")
        outfile.write(f"\nFrase con probabilità di Markov più alta: \n{max_markov_sentence} ({max_markov_probability})\n")
        outfile.write(f"\nPercentuale di Stopwords: \n{stopwords_percentage}\n")
        outfile.write(f"\nNumero di pronomi personali: \n{personal_pronouns_count}\n")
        outfile.write(f"\nMedia di pronomi personali per frase: \n{avg_personal_pronouns}\n")
        outfile.write(f"\nTop-15 Named Entities: \n{top_named_entities}\n")

    
if __name__ == "__main__":
    # Definisco i percorsi dei file
    INPUT_FILE1 = "corpus1.txt"
    INPUT_FILE2 = "corpus2.txt"

    # Eseguo la funzione principale
    main(INPUT_FILE1)
    main(INPUT_FILE2)