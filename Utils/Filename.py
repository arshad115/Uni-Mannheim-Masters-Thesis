import settings
import os
class Filename:
    def __init__(self, filename):
        # filename = 'subset-tokens'
        DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
        OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY")
        self.text_filename = DATA_DIRECTORY + filename + '.txt'
        self.corpus_filename = OUTPUT_DIRECTORY + 'models/' + filename + '.pkl'
        self.dictionary_filename = OUTPUT_DIRECTORY + filename + '.gensim'
        self.model_filename = OUTPUT_DIRECTORY + 'models/' + filename + '.gensim'
        self.tokens_filename = DATA_DIRECTORY + filename + '-tokens.txt'
        self.tokens_output_filename = OUTPUT_DIRECTORY + 'tokens/' + filename + '-tokens.txt'
        self.nq_filename = DATA_DIRECTORY + filename + '.nq'
        self.raw_concept_filename = OUTPUT_DIRECTORY  + 'concepts/raw/' + filename + '.txt'
        self.clean_concept_filename = OUTPUT_DIRECTORY + 'concepts/clean/' + filename + '.txt'
        self.split_nq_filename = DATA_DIRECTORY + 'splits/' + filename + '.nq'
        self.clean_concepts_path = OUTPUT_DIRECTORY + 'concepts/subset/'
        self.raw_concepts_path = OUTPUT_DIRECTORY + 'concepts/raw/'
