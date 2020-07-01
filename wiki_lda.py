import logging
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
import settings
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY")

dictionary = gensim.corpora.Dictionary.load_from_text(os.path.join(OUTPUT_DIRECTORY ,'wikipedia','wiki_wordids.txt'))
corpus = os.path.join(OUTPUT_DIRECTORY ,'wikipedia','wiki_tfidf.mm')

if __name__ == '__main__':
    for num_topics in range(10, 1000, 10):
        print(f'Training model with {num_topics} topics...')
        model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=5)
        modelFile = os.path.join(OUTPUT_DIRECTORY, 'models', 'wiki', 'lda_' + str(num_topics) + '.gensim')
        model.save(modelFile)
        print('Model saved...')