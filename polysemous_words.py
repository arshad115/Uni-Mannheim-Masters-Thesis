from typing import List, Tuple
import gensim
import jsonpickle
from gensim.models import LdaMulticore
from evaluate import evaluate
import skopt
import regex
import time
import settings
import os
import csv
import neptune
import neptunecontrib.monitoring.skopt as sk_utils

DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY")

dictionary = gensim.corpora.Dictionary.load(os.path.join(OUTPUT_DIRECTORY, 'models', 'concept', 'lda_dict.dict'))
lda = LdaMulticore.load(os.path.join(OUTPUT_DIRECTORY, 'models', 'concept', 'lda_10.gensim'))

ACCURACY = 1
PRECISION = 2
RECALL = 3
F1 = 4
MEASURE_STR = ''

# Change the measure here:
MEASURE = F1
CALLS = 500
RANDOM_CALLS = 50

lda_model_topics = 10

PROB_MIN = 0.05
PROB_MAX = 0.1

T_MIN = 1
T_MAX = 5

MAX_DOC_WORDS = 1706800  # All
MIN_DOC_WORDS = 5

# If true then hyper parameter optimization will be done for the space params
SCIKIT_OPTIMIZE = False

# Place parameters in space to optimize, replace the usages accordingly
if SCIKIT_OPTIMIZE:
    SPACE = [
        skopt.space.Integer(MIN_DOC_WORDS, 500, name='MIN_DOC_LEN'),
        skopt.space.Real(PROB_MIN, PROB_MAX, name='minimum_probability', prior='uniform'),
        skopt.space.Integer(T_MIN, T_MAX, name='minimum_topics_polysemeous'),
    ]
else:
    SPACE = {'MIN_DOC_LEN': MIN_DOC_WORDS,
             'minimum_probability': 0.0575,
             'minimum_topics_polysemeous': 2
             }

conceptsDir = os.path.join(OUTPUT_DIRECTORY, 'concepts', 'clean')

topic_probs = []
cleanConceptNames = {}
conceptPolysemy = {}
files = []
conceptInWikiOrDisambiguation = {}

conceptTexts = {}
conceptsLoaded = []


def cleanString(str):
    remove = regex.compile(r'[\p{C}|\p{M}|\p{P}|\p{S}|\p{Z}]+', regex.UNICODE)
    str = remove.sub(u" ", str).strip()
    str = str.lower()
    return str


def getConceptTopics(concept, search_params):
    bow = dictionary.doc2bow(conceptTexts[concept])
    topics = lda.get_document_topics(bow, minimum_probability=search_params[
        'minimum_probability'])  # =search_params['minimum_probability']
    topics = sorted(topics, key=lambda tup: tup[1], reverse=True)
    cleanName = cleanConceptNames[concept]
    conceptPolysemy[cleanName] = True if len(topics) > search_params['minimum_topics_polysemeous'] else False
    conceptsLoaded.append(cleanName)


def loadConceptFile(concept):
    file = os.path.join(conceptsDir, concept)
    if (os.path.isfile(file)):
        doc = open(file, encoding="utf8").read()
        text = doc.split()
        conceptTexts[concept] = text


def loadAllConceptsInMemory():
    for concept in cleanConceptNames.keys():
        loadConceptFile(concept)


def runAllConcepts(search_params):
    global conceptsLoaded
    conceptsLoaded = []
    for concept in cleanConceptNames.keys():
        if (len(conceptTexts[concept]) > MIN_DOC_WORDS and len(conceptTexts[concept]) <= MAX_DOC_WORDS):
            getConceptTopics(concept, search_params)
    print(f'Total docs loaded: {len(conceptsLoaded)}')
    neptune.send_metric('TOTAL_DOCS', len(conceptsLoaded))


def saveBow():
    json_string = jsonpickle.encode(conceptTexts)
    jsonFile = os.path.join(DATA_DIRECTORY, 'evaluation', 'wiki', 'conceptTexts.json')
    with open(jsonFile, 'w') as json_file:
        json_file.write(json_string)
        json_file.close()


def saveJson(search_params):
    json_string = jsonpickle.encode(conceptPolysemy)
    jsonFile = os.path.join(OUTPUT_DIRECTORY, 'poly',
                            'polysemeous_' + str(search_params['minimum_topics_polysemeous']) + '_' +
                            str(search_params['minimum_probability']) + '.json')  # search_params['minimum_probability']
    with open(jsonFile, 'w') as json_file:
        json_file.write(json_string)
        json_file.close()


def getconceptTextsBow():
    filename = os.path.join(DATA_DIRECTORY, 'evaluation', 'wiki', 'conceptTexts.json')
    jsondata = open(filename, encoding='utf-8').read()
    thawed = jsonpickle.decode(jsondata)
    return thawed


def getCleanConceptNames():
    filename = os.path.join(DATA_DIRECTORY, 'evaluation', 'wiki', 'cleanConceptNames.json')
    jsondata = open(filename, encoding='utf-8').read()
    thawed = jsonpickle.decode(jsondata)
    return thawed


def getWikiDisambiguationPages():
    filename = os.path.join(DATA_DIRECTORY, 'evaluation', 'wiki', 'wikiDisambiguationPages.json')
    jsondata = open(filename, encoding='utf-8').read()
    thawed = jsonpickle.decode(jsondata)
    return thawed


def getConceptInWikiOrDisambiguation():
    filename = os.path.join(DATA_DIRECTORY, 'evaluation', 'wiki', 'conceptInWikiOrDisambiguation.json')
    jsondata = open(filename, encoding='utf-8').read()
    thawed = jsonpickle.decode(jsondata)
    return thawed


def getAllWikis():
    filename = os.path.join(DATA_DIRECTORY, 'evaluation', 'wiki', 'conceptWikis.json')
    jsondata = open(filename, encoding='utf-8').read()
    thawed = jsonpickle.decode(jsondata)
    return thawed


def saveEVRow(evResult, search_params):
    experiment_file = 'evresults_' + MEASURE_STR + '_' + str(PROB_MIN) + '_' + str(PROB_MAX) + '_' + str(
        T_MIN) + '_' + str(T_MAX) + '.csv'
    filename = os.path.join(OUTPUT_DIRECTORY, 'evaluation', experiment_file)
    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(filename, append_write, encoding='utf-8', newline='') as ev_file:
        ev_writer = csv.writer(ev_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        ev_writer.writerow([search_params['minimum_probability'], search_params['minimum_topics_polysemeous'],
                            evResult.true_positives, evResult.false_positives, evResult.true_negatives,
                            evResult.false_negatives,
                            evResult.getPrecision(), evResult.getRecall(), evResult.getAccuracy(),
                            evResult.getF1Score()])


def sendMetrics(evResult, search_params):
    global lda_model_topics

    neptune.send_metric('lda_model_topics', lda_model_topics)
    neptune.send_metric('minimum_probability', search_params['minimum_probability'])
    neptune.send_metric('minimum_topics_polysemeous', search_params['minimum_topics_polysemeous'])

    neptune.send_metric('true_positives', evResult.true_positives)
    neptune.send_metric('false_positives', evResult.false_positives)
    neptune.send_metric('true_negatives', evResult.true_negatives)
    neptune.send_metric('false_negatives', evResult.false_negatives)
    neptune.send_metric('precision', evResult.getPrecision())
    neptune.send_metric('recall', evResult.getRecall())
    neptune.send_metric('accuracy', evResult.getAccuracy())
    neptune.send_metric('f1', evResult.getF1Score())


def train_evaluate(search_params):
    mind = search_params['MIN_DOC_LEN']

    print(f'Running on all concepts with min doc len: {mind}\n', flush=True)
    start_time = time.time()

    neptune.send_metric('MIN_DOC_LEN', mind)

    runAllConcepts(search_params)

    print('Finished Processing concepts, --- %s minutes ---' % ((time.time() - start_time) / 60), flush=True)
    print('Evaluating\n', flush=True)
    start_time = time.time()

    # if conceptInWikiOrDisambiguation is not found then create with evaluate.saveConceptInWikiOrDisambiguation()
    evResult = evaluate(conceptsLoaded, conceptPolysemy, conceptInWikiOrDisambiguation)

    try:
        saveEVRow(evResult)
        sendMetrics(evResult)
    except:
        print('Error sending metrics', evResult)

    # saveJson()

    print('Finished evaluating concepts, --- %s minutes ---' % ((time.time() - start_time) / 60), flush=True)

    if MEASURE == ACCURACY:
        return evResult.getAccuracy()
    elif MEASURE == PRECISION:
        return evResult.getPrecision()
    elif MEASURE == RECALL:
        return evResult.getRecall()
    elif MEASURE == F1:
        return evResult.getF1Score()
    else:
        return evResult.getAccuracy()


if SCIKIT_OPTIMIZE:
    @skopt.utils.use_named_args(SPACE)
    def objective(**params):
        result = train_evaluate(params)
        print('Result: ' + str(result) + ' Params: ' + str(params['minimum_probability']) + ', ' + str(
            params['minimum_topics_polysemeous']))
        return -1.0 * result

if __name__ == '__main__':
    start_time = time.time()

    if MEASURE == ACCURACY:
        MEASURE_STR = 'accuracy'
    elif MEASURE == PRECISION:
        MEASURE_STR = 'precision'
    elif MEASURE == RECALL:
        MEASURE_STR = 'recall'
    elif MEASURE == F1:
        MEASURE_STR = 'f1'
    else:
        MEASURE_STR = 'accuracy'

    neptune.set_project('arshad115/thesis')
    experiment_name = 'LDA_' + MEASURE_STR + '_' + str(CALLS) + '_' + str(PROB_MIN) + '_' + str(PROB_MAX) + '_' + str(
        T_MIN) + '_' + str(T_MAX)
    neptune.create_experiment(name=experiment_name, upload_source_files=[])

    print('Loading files into the memory\n', flush=True)
    conceptInWikiOrDisambiguation = getConceptInWikiOrDisambiguation()
    cleanConceptNames = getCleanConceptNames()
    loadAllConceptsInMemory()

    print('Files loaded, --- %s minutes ---' % ((time.time() - start_time) / 60), flush=True)

    if SCIKIT_OPTIMIZE:
        monitor = sk_utils.NeptuneMonitor()
        results = skopt.forest_minimize(objective, SPACE, n_calls=CALLS, n_random_starts=RANDOM_CALLS,
                                        callback=[monitor])
        best_auc = -1.0 * results.fun
        best_params = results.x

        print('Finished processing everything, --- %s minutes ---' % ((time.time() - start_time) / 60), flush=True)

        print('best result: ', best_auc, flush=True)
        print('best parameters: ', best_params, flush=True)

    else:
        result = train_evaluate(SPACE)
        print('Result: ' + str(result) + ' LDA-Topics: ' + str(lda_model_topics) + ' Params: ' + str(
            SPACE['minimum_probability']) + ', ' + str(SPACE['minimum_topics_polysemeous']))

    print('Finished processing everything, --- %s minutes ---' % ((time.time() - start_time) / 60), flush=True)
    neptune.stop()
