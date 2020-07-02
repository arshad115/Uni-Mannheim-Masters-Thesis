import csv
import string
from typing import List, Tuple
import jsonpickle
from strsimpy.jaro_winkler import JaroWinkler
import regex
import settings
import os
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY")

# threshold = 0.03
# minimum_probability = 0.03555
# minimum_topics_polysemeous = 3
#
# jsonFile = OUTPUT_DIRECTORY + 'poly/polysemeous_' + str(minimum_topics_polysemeous) + '_'+ str(threshold) + '_' + str(minimum_probability) +'.json'

conceptsDir = os.path.join(OUTPUT_DIRECTORY ,'concepts','clean')
jarowinkler = JaroWinkler()

class EVResult:
    def __init__(self,true_positives, false_positives, true_negatives, false_negatives):
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.true_negatives = true_negatives
        self.false_negatives = false_negatives
        self.precision = 0
        self.f1 = 0
        self.recall = 0
        self.accuracy = 0

    def getPrecision(self):
        try:
            precision = (self.true_positives) / (self.true_positives + self.false_positives)
            self.precision = precision
            return self.precision
        except:
            return 0

    def getRecall(self):
        try:
            recall = (self.true_positives) / (self.true_positives + self.false_negatives)
            self.recall = recall
            return self.recall
        except:
            return 0

    def getAccuracy(self):
        try:
            accuracy = (self.true_positives + self.true_negatives) / (
                    self.true_positives + self.true_negatives + self.false_negatives + self.false_positives)
            self.accuracy = accuracy
            return self.accuracy
        except:
            return 0

    def getF1Score(self):
        try:
            f1 = (2 * self.getPrecision() * self.getRecall() / (self.getPrecision() + self.getRecall()))
            self.f1 = f1
            return self.f1
        except:
            return 0

    def printEvaluation(self):
        print('\n', flush=True)
        print('True positives: ' + str(self.true_positives), flush=True)
        print('False negatives: ' + str(self.false_negatives), flush=True)
        print('False positives: ' + str(self.false_positives), flush=True)
        print('True negatives: ' + str(self.true_negatives), flush=True)

        print('Precision: ' + str(self.getPrecision()), flush=True)
        print('Recall: ' + str(self.getRecall()), flush=True)
        print('Accuracy: ' + str(self.getAccuracy()), flush=True)
        print('F1 Score: ' + str(self.getF1Score()), flush=True)
        # return self

def getWikiDisambiguationPages():
    filename = DATA_DIRECTORY + 'evaluation\wiki\disambiguation.csv'
    titles = []
    pageId = []
    with open(filename, encoding='utf-8') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            titles.append(row[1])
    return titles

def getCleanConceptNames():
    filename = os.path.join(DATA_DIRECTORY, 'evaluation', 'wiki', 'cleanConceptNames.json')
    jsondata = open(filename, encoding='utf-8').read()
    thawed = jsonpickle.decode(jsondata)
    return thawed

def cleanString(str):
    remove = regex.compile(r'[\p{C}|\p{M}|\p{P}|\p{S}|\p{Z}]+', regex.UNICODE)
    str = remove.sub(u" ", str).strip()
    str = str.lower()
    return str

def compareStrings(str1, str2):
    sim = jarowinkler.similarity(str1,str2)
    print(sim)
    if sim >= 0.95:
        return True
    else:
        return False

def getAllWikis():
    filename = DATA_DIRECTORY + 'evaluation\wiki\enwiki-latest-all-titles-in-ns0'
    wikis = set(line.strip() for line in open(filename, encoding='utf-8'))
    return wikis

def saveCleanNames():
    print('Saving new cleanConceptNames file...')
    files = os.listdir(conceptsDir)
    conceptNames = {}
    for concept in files:
        conceptNames[concept] = cleanString(concept.replace('.txt', ''))
    json_string = jsonpickle.encode(conceptNames)
    jsonFile = os.path.join(DATA_DIRECTORY, 'evaluation', 'wiki', 'cleanConceptNames.json')
    with open(jsonFile, 'w') as json_file:
        json_file.write(json_string)
        json_file.close()

def saveConceptInWikiOrDisambiguation():
    cleanConceptNames = getCleanConceptNames().values() # if not found then create with saveCleanNames()

    wikiDisambiguationPages = getWikiDisambiguationPages()
    disambiguationPages = set(wikiDisambiguationPages)

    wikiset = getAllWikis()

    # common = wikiset.intersection(disambiguationPages)

    wikisWithoutDisambiguationPages = wikiset - disambiguationPages

    cleanwikipages = set(cleanString(val) for val in wikisWithoutDisambiguationPages)
    conceptWikis = cleanwikipages.intersection(cleanConceptNames)

    index = 0
    while index < len(wikiDisambiguationPages):
        wiki = wikiDisambiguationPages[index]
        wiki = wiki.replace('(disambiguation)', '')
        wiki = cleanString(wiki)
        wikiDisambiguationPages[index] = wiki
        index += 1

    filename = DATA_DIRECTORY + 'evaluation\wiki\wikiDisambiguationPages.json'
    json_string = jsonpickle.encode(wikiDisambiguationPages)
    with open(filename, 'w') as json_file:
        json_file.write(json_string)
        json_file.close()

    filename = DATA_DIRECTORY + 'evaluation\wiki\conceptWikis.json'
    json_string = jsonpickle.encode(conceptWikis)
    with open(filename, 'w') as json_file:
        json_file.write(json_string)
        json_file.close()

    conceptInWikiOrDisambiguation = {}

    for concept in cleanConceptNames:
        if concept in wikiDisambiguationPages:
            conceptInWikiOrDisambiguation[concept] = 1
        elif concept in conceptWikis:
            conceptInWikiOrDisambiguation[concept] = 2
        else:
            conceptInWikiOrDisambiguation[concept] = 0

    filename = DATA_DIRECTORY + 'evaluation\wiki\conceptInWikiOrDisambiguation.json'
    json_string = jsonpickle.encode(conceptInWikiOrDisambiguation)
    with open(filename, 'w') as json_file:
        json_file.write(json_string)
        json_file.close()

def evaluate(concepts, conceptPolysemy, conceptInWikiOrDisambiguation):

    total = len(concepts)
    true_positives = 0
    false_negatives = 0
    matches_found = 0
    matches_found_in_wikis = 0
    false_positives = 0
    true_negatives = 0
    totalpolysemeous = 0
    totalnonpolysemeous = 0

    for concept in concepts:
        if conceptInWikiOrDisambiguation[concept] == 1:
            if conceptPolysemy[concept] == True :
                true_positives += 1
                totalpolysemeous += 1
            else:
                false_negatives += 1
                totalnonpolysemeous += 1
            matches_found += 1
        elif conceptInWikiOrDisambiguation[concept] == 2:
            if conceptPolysemy[concept] == True:
                false_positives += 1
                totalpolysemeous += 1
            else:
                true_negatives += 1
                totalnonpolysemeous += 1
            matches_found_in_wikis += 1

    print('Total Concepts: ' + str(total),flush=True)
    print('Total polysemeous: ' + str(totalpolysemeous),flush=True)
    print('Total non polysemeous: ' + str(totalnonpolysemeous),flush=True)
    print('Matches found in disambiguation pages: ' + str(matches_found),flush=True)
    print('Matches found in Wikipedia pages: ' + str(matches_found_in_wikis),flush=True)
    print('Total matches found: ' + str(matches_found + matches_found_in_wikis),flush=True)

    evresult = EVResult(true_positives, false_positives, true_negatives, false_negatives)
    evresult.printEvaluation()
    return evresult

if __name__ == '__main__':
    saveConceptInWikiOrDisambiguation()
