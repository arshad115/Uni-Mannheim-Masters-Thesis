import logging
import os
import settings
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY")

dict_file = os.path.join(OUTPUT_DIRECTORY, 'models', 'concept', 'lda_dict.dict')
conceptsDir = os.path.join(OUTPUT_DIRECTORY, 'concepts', 'clean')

train_corpus = []

def loadConceptFile(concept):
    file = os.path.join(conceptsDir, concept)
    doc = open(file, encoding="utf8").read()
    return doc


def loadAllConceptFiles():
    files = os.listdir(conceptsDir)
    for concept in files:
        text = loadConceptFile(concept)
        # Add document
        if (len(text) > MIN_DOC_WORDS and len(text) <= MAX_DOC_WORDS):
            train_corpus.append(text.split())


MAX_DOC_WORDS = 1706800  # All
MIN_DOC_WORDS = 5

if __name__ == '__main__':
    print('Loading files...')
    loadAllConceptFiles()
    print('Files loaded...')

    # Create a corpus from a list of texts
    dictionary = Dictionary(train_corpus)
    # dictionary.filter_extremes(no_above=0.8, no_below=3)

    dictionary.compactify()  # Reindexes the remaining words after filtering
    print("Left with {} words.".format(len(dictionary.values())))

    # Save the Dictionary for later use
    dictionary.save(dict_file)

    corpus = [dictionary.doc2bow(text) for text in train_corpus]

    # Train the model on the corpus.
    for num_topics in range(10, 1000, 10):
        print(f'Training model with {num_topics} topics...')
        model = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=5)
        modelFile = os.path.join(OUTPUT_DIRECTORY, 'models', 'concept', 'lda_' + str(num_topics) + '.gensim')
        model.save(modelFile)
        print('Model saved...')
