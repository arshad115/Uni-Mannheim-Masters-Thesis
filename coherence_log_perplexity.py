import time

import gensim
import nltk
from gensim.models import CoherenceModel, LdaModel, HdpModel
import settings
import os

DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY")

# Replace with model specific dictionary and corpus
dictionary = gensim.corpora.Dictionary.load_from_text(os.path.join(OUTPUT_DIRECTORY ,'wikipedia','wiki_wordids.txt'))
corpus = os.path.join(OUTPUT_DIRECTORY ,'wikipedia','wiki_tfidf.mm')

files = []
texts = []

conceptsDir = os.path.join(OUTPUT_DIRECTORY ,'concepts','dsubset')

def loadConceptFiles():
    for concept in files:
        text = preprocess(loadConceptFile(concept))
        texts.append(text)

# Tokenize
def preprocess(text):
    result = nltk.word_tokenize(text)
    return result

def loadConceptFile(concept):
    file = os.path.join(conceptsDir, concept)
    doc = open(file, encoding="utf8").read()
    return doc

def compute_coherence_values(limit, start=10, step=10):
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel.load(os.path.join(OUTPUT_DIRECTORY, 'models','concept', f'lda_{num_topics}.gensim'))

        print('Model topics:', str(num_topics))
        lp = model.log_perplexity(corpus)

        print('Perplexity:', str(lp))

        # Commented because takes too long
        # coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        # c1 = coherencemodel.get_coherence()
        # print('Coherence Score(c_v):', str(c1))

        coherence_model_lda = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        c2 = coherence_model_lda.get_coherence()
        print('Coherence Score(u_mass):', str(c2))


        print('\nTopics:', num_topics, 'Perplexity:', lp, 'Coherence Score(u_mass):', c2)


if __name__ == '__main__':
    start_time = time.time()

    # files = os.listdir(conceptsDir)
    # loadConceptFiles()

    compute_coherence_values(620)

    print('Completed processing everything, --- %s minutes ---' % ((time.time() - start_time) / 60),flush=True)