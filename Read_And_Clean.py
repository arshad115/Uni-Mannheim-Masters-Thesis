from rdflib import Graph, URIRef, ConjunctiveGraph
from Utils.Filename import Filename
from Utils.Constants import Constants
from pathlib import Path
import time
import os
import re
import regex
import nltk
from spellchecker import SpellChecker
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

def removeNumbers(list):
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in list]
    return list

def removeMistakes(text):
    spell = SpellChecker()
    # find those words that may be misspelled
    misspelled = spell.unknown(text)
    nomistakes = set(text) - set(misspelled)
    return nomistakes

def removeNonEnglish(text):
    cleaned = " ".join(w for w in text if w.lower() in words or not w.isalpha())
    cleaned = cleaned.split()
    return cleaned

def tokenize(text):
    result = nltk.word_tokenize(text)
    return result

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def toLower(text):
    return text.lower()

def removePunctuation(list):
    remove = regex.compile(r'[\p{C}|\p{M}|\p{P}|\p{S}|\p{Z}]+', regex.UNICODE)
    text = remove.sub(u" ", list).strip()
    return text

def processTextForLDA(text):

    # Lower case
    text = toLower(text)

    # Remove punctuation
    text = removePunctuation(text)

    # Tokenize
    tokens = tokenize(text)

    # Remove numbers
    tokens = removeNumbers(tokens)

    # Remove stop words
    tokens = [token for token in tokens if token not in en_stop]

    # Lemmatization
    tokens = [get_lemma(token) for token in tokens]

    # Remove small tokens
    tokens = [token for token in tokens if len(token) > 2]

    # Remove mistakes
    tokens = removeMistakes(tokens)

    # Remove non-English words
    tokens = removeNonEnglish(tokens)

    tokens = [token for token in tokens if token]

    cleanText = " ".join(tokens)

    return cleanText

def writeFiles(filename, rawText, cleanText):

    with open(filename.raw_concept_filename, 'a', encoding="utf8") as f:
        f.write('\n'.join(rawText))
        f.write('\n')

    # Write clean file
    with open(filename.clean_concept_filename, 'a', encoding="utf8") as f:
        f.write('\n'.join(cleanText))
        f.write('\n')


def processFile(filenameSubset):
    try:
        global total_concepts
        start_time = time.time()

        os.makedirs(os.path.dirname(filenameSubset.raw_concept_filename), exist_ok=True)
        os.makedirs(os.path.dirname(filenameSubset.clean_concept_filename), exist_ok=True)

        g = ConjunctiveGraph()
        data = open(filenameSubset.nq_filename, "rb")
        g.parse(data, format="nquads")

        ### Get All Contexts
        i = 0
        for ctx in g.contexts():
            i = i + 1

            concept_url = ''
            concept = ''

            for subject, predicate, obj in ctx:
                concept_url = subject
                concept = str(concept_url).replace(Constants.ConceptPrefix, '')
                break

            node_identifier = ctx.identifier

            cleanTexts = []
            rawTexts = []

            if (str(concept_url).startswith(Constants.ConceptPrefix)):
                # create file for concept
                filename = Filename(str(concept))

                path = Path(filename.raw_concept_filename)
                path.parent.mkdir(parents=True, exist_ok=True)

                path = Path(filename.clean_concept_filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                
                wasDerivedFrom_objects = g.objects(subject=node_identifier, predicate=URIRef(Constants.wasDerivedFrom))

                for wasDerivedFrom in wasDerivedFrom_objects:
                    provValues = g.objects(subject=wasDerivedFrom, predicate=URIRef(Constants.provValue))

                    for provValue in provValues:
                        rawTexts.append(provValue)

                        clean_text = processTextForLDA(provValue)
                        cleanTexts.append(clean_text)

            writeFiles(filename, rawTexts, cleanTexts)
           
        total_concepts = total_concepts + i
        print('Finished Processing: ' + filenameSubset.nq_filename + ' ,Concepts:' + str(i) + ' ,Total Concepts:' + str(total_concepts) + ", --- %s seconds ---" % (time.time() - start_time))
    except Exception as e:
        print('Error processing File: ' + filenameSubset.nq_filename + ' ,' + str(e))


total_concepts = 0
en_stop = None
words = None

if __name__ == '__main__':
    start_time = time.time()

    nltk.download('punkt')
    nltk.download('words')

    filename = Filename('subset-1')

    en_stop = set(nltk.corpus.stopwords.words('english'))
    words = set(nltk.corpus.words.words())

    try:
        processFile(filename)
    except:
        print('Error processing File: ' + filename.nq_filename)

    print('Concept files cleaned and saved, --- %s minutes ---' % ((time.time() - start_time) / 60), flush=True)