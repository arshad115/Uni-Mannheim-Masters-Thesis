import logging
import gensim
from gensim.models import HdpModel
import settings
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY")

ldamodelFile = os.path.join(OUTPUT_DIRECTORY, 'models', 'wiki', 'lda_hdp_wiki.gensim')
hdpmodelFile = os.path.join(OUTPUT_DIRECTORY, 'models', 'wiki', 'hdp_wiki.gensim')

dictionary = gensim.corpora.Dictionary.load_from_text(os.path.join(OUTPUT_DIRECTORY ,'wikipedia','wiki_wordids.txt'))
corpus = os.path.join(OUTPUT_DIRECTORY ,'wikipedia','wiki_tfidf.mm')

print("training hdp model")

hdpmodel = HdpModel(corpus=corpus, id2word=dictionary, T=1000)
hdpmodel.save(hdpmodelFile)
print("saved hdp model\n")

# Also save the suggested lda model - lda model which closely resembles the hdp model
print("saving suggested lda model\n")
lda = hdpmodel.suggested_lda_model()
lda.save(ldamodelFile)

print("suggested lda model saved!\n")
