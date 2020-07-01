import csv
import logging
import math
import os
import os
import pickle
import re
import time
import warnings
from pprint import pprint
from typing import List, Tuple
import gensim
import gensim
import gensim.corpora as corpora
import jsonpickle
import matplotlib.pyplot as plt
import neptune
import neptunecontrib.monitoring.skopt as sk_utils
import nltk
import numpy as np
import pandas as pd
import regex
import skopt
from gensim.models import CoherenceModel, LdaMulticore, HdpModel
from gensim.utils import simple_preprocess
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from nltk.corpus import wordnet
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from spellchecker import SpellChecker
import settings
import os

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY")

# Concepts LDA
dictionary = gensim.corpora.Dictionary.load(os.path.join(OUTPUT_DIRECTORY, 'models', 'concept', 'lda_dict.dict'))
lda = gensim.models.ldamodel.LdaModel.load(os.path.join(OUTPUT_DIRECTORY, 'models', 'concept', 'lda_10.gensim'))

# # Wiki LDA
# dictionary = gensim.corpora.Dictionary.load_from_text(os.path.join(OUTPUT_DIRECTORY ,'wikipedia','wiki_wordids.txt'))
# corpus = gensim.corpora.MmCorpus(os.path.join(OUTPUT_DIRECTORY ,'wikipedia','wiki_tfidf.mm'))
# lda = gensim.models.ldamodel.LdaModel.load(os.path.join(OUTPUT_DIRECTORY, 'models', 'wiki','lda_10.gensim'))

conceptsDir = os.path.join(OUTPUT_DIRECTORY, 'concepts', 'clean')

train_corpus = []
train_corpus_excluded = []
train_corpus_splitted = []
conceptPolysemyGS = {}
# wikiDisambiguationPages = {}
ytrainwiki = []
ytrainwiki_excluded = []
conceptInWikiOrDisambiguation = {}
cleanConceptNames = {}


def cleanString(str):
    remove = regex.compile(r'[\p{C}|\p{M}|\p{P}|\p{S}|\p{Z}]+', regex.UNICODE)
    str = remove.sub(u" ", str).strip()
    str = str.lower()
    return str


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


def getCleanConceptNames():
    filename = os.path.join(DATA_DIRECTORY, 'evaluation', 'wiki', 'cleanConceptNames.json')
    jsondata = open(filename, encoding='utf-8').read()
    thawed = jsonpickle.decode(jsondata)
    return thawed


def loadAllConceptFiles(search_params):
    global train_corpus, ytrainwiki, train_corpus_excluded, ytrainwiki_excluded
    train_corpus = []
    ytrainwiki = []
    train_corpus_excluded = []
    ytrainwiki_excluded = []
    for concept in cleanConceptNames.keys():
        rawtext = loadConceptFile(concept)
        text = rawtext.split()

        cleanName = cleanConceptNames[concept]

        if ADD_NAME_IN_DOC:
            text = cleanName + ' ' + text

        if (len(text) > search_params['MIN_DOC_LEN'] and len(text) <= MAX_DOC_WORDS):
            train_corpus.append(text)

            # train_corpus_splitted.append(text.split())
            if conceptInWikiOrDisambiguation[cleanName] == 1:
                ytrainwiki.append(1)
            else:
                ytrainwiki.append(0)
        else:
            train_corpus_excluded.append(text)
            if conceptInWikiOrDisambiguation[cleanName] == 1:
                ytrainwiki_excluded.append(1)
            else:
                ytrainwiki_excluded.append(0)
    print(f'Documents in train_corpus: {len(train_corpus)}')
    print(f'Documents in train_corpus_excluded: {len(train_corpus_excluded)}')


def loadConceptFile(concept):
    file = os.path.join(conceptsDir, concept)
    doc = open(file, encoding="utf8").read()
    return doc


def getSingleVector(textSplits):
    bow = dictionary.doc2bow(textSplits)
    top_topics = lda.get_document_topics(bow, minimum_probability=0.0)
    topic_vec = [x[1] for x in top_topics]
    if SORT_TOPICS:
        topic_vec = sorted(topic_vec, reverse=True)
    if ADD_LENGTH_FEATURE:
        topic_vec.extend([len(textSplits)])  # length concept text

    return topic_vec


def getTrainVecs(train_corpus):
    train_vecs = []

    if REMOVE_DUPLICATE_DOCS:
        train_corpus = list(set(train_corpus))

    for i in range(len(train_corpus)):
        text = train_corpus[i]
        topic_vec = getSingleVector(text)
        train_vecs.append(topic_vec)

    print(f'Length of features {len(train_vecs[0])}')
    return train_vecs


def runLogisticRegression(X_train_scale, X_val_scale, y_train, y_test):
    print("Training a Logistic Regression Model...")

    scikit_log_reg = LogisticRegression(
        class_weight='balanced',
        solver='saga',
        max_iter=10000,
        C=0.1,
        fit_intercept=True)

    model = scikit_log_reg.fit(X_train_scale, y_train)
    y_pred = model.predict(X_val_scale)

    f1 = f1_score(y_test, y_pred, average='binary')
    if NEPTUNE_SEND:
        neptune.send_metric('LR-f1', f1)
    print('Logistic Regression f1:', f1)

    if TEST_EXCLUDED:
        testModelonExcluded(model, 'LR')

    return f1


def runModelBayesSearchCV(X_train_scale, X_val_scale, y_train, y_test):
    print("Fine tuning Model...")

    # Change model and params here for the model you want to fine tune
    clf = LogisticRegression(max_iter=10000)

    # this is our parameter grid
    param_grid = {
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced'],
        'penalty': ['l1', 'l2'],
        'tol': Real(1e-5, 1e-3, 'log-uniform'),
        'C': Real(1e-5, 100, 'log-uniform'),
        'fit_intercept': [True, False]
    }

    # set up our optimiser to find the best params in 30 searches
    opt = BayesSearchCV(
        clf,
        param_grid,
        scoring='f1',
        n_iter=30,
        random_state=1234,
        verbose=5
    )

    print(opt.total_iterations)

    opt.fit(X_train_scale, y_train)

    print('Best params achieve a test score of', opt.score(X_val_scale, y_test), ':')

    print(opt.best_params_)

    print('Best params achieve a test score of', opt.best_estimator_, ':')


def runSGDClassifier(X_train_scale, X_val_scale, y_train, y_test):
    print("Training a Logistic Regression SGD Model...")
    sgd = linear_model.SGDClassifier(
        max_iter=10000,
        tol=1e-3,
        loss='log',
        class_weight='balanced'
    ).fit(X_train_scale, y_train)
    y_pred = sgd.predict(X_val_scale)

    f1 = f1_score(y_test, y_pred, average='binary')

    if NEPTUNE_SEND:
        neptune.send_metric('LRSGD-f1', f1)
    print('Logistic Regression SGD f1:', f1)

    if TEST_EXCLUDED:
        testModelonExcluded(sgd, 'LRSGD')

    return f1


def runSGDHuberClassifier(X_train_scale, X_val_scale, y_train, y_test):
    print("Training a SGD Modified Huber Model...")
    sgd_huber = linear_model.SGDClassifier(
        max_iter=10000,
        tol=1e-3,
        alpha=1,
        loss='modified_huber',
        class_weight='balanced'
    ).fit(X_train_scale, y_train)
    y_pred = sgd_huber.predict(X_val_scale)

    f1 = f1_score(y_test, y_pred, average='binary')

    if NEPTUNE_SEND:
        neptune.send_metric('LRSGDH-f1', f1)
    print('SGD Modified Huber f1:', f1)

    if TEST_EXCLUDED:
        testModelonExcluded(sgd_huber, 'LRSGDH')

    return f1


def runAllSVM(X_train_scale, X_val_scale, y_train, y_test):
    kernels = ['linear', 'rbf', 'poly']
    gammas = [0.1, 1, 10, 100]
    degrees = [0, 1, 2, 3, 4, 5, 6]

    cv_svclinear_f1 = []

    for kernel in kernels:
        if kernel != 'linear':
            for gamma in gammas:
                if kernel == 'poly':
                    for degree in degrees:
                        svm = runSVMClassifier(X_train_scale, X_val_scale, y_train, y_test, kernel, degree=degree,
                                               gamma=gamma)
                        cv_svclinear_f1.append(svm)
                else:
                    svm = runSVMClassifier(X_train_scale, X_val_scale, y_train, y_test, kernel, gamma=gamma)
                    cv_svclinear_f1.append(svm)
        else:
            svm = runSVMClassifier(X_train_scale, X_val_scale, y_train, y_test, kernel)
            cv_svclinear_f1.append(svm)
    return cv_svclinear_f1


def runSVMClassifier(X_train_scale, X_val_scale, y_train, y_test, kernel='linear', gamma='scale', degree=3):
    print(f'Training a SVM {kernel}, {gamma}, {degree} Model...')
    # Create a svm Classifier
    clf = SVC(kernel=kernel, gamma=gamma, degree=degree)  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train_scale, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_val_scale)

    f1 = f1_score(y_test, y_pred, average='binary')

    if NEPTUNE_SEND:
        neptune.send_metric(f'SVM-{kernel}-{gamma}-{degree}-f1', f1)
    print(f'SVM {kernel}, {gamma}, {degree}:', f1)

    return clf


def runGaussianNBClassifier(X_train_scale, X_val_scale, y_train, y_test):
    print(f'Training a GaussianNB Model...')
    clf = GaussianNB()

    # Train the model using the training sets
    clf.fit(X_train_scale, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_val_scale)

    f1 = f1_score(y_test, y_pred, average='binary')

    if NEPTUNE_SEND:
        neptune.send_metric(f'GaussianNB-f1', f1)
    print(f'GaussianNB:', f1)

    if TEST_EXCLUDED:
        testModelonExcluded(clf, 'GaussianNB')

    return f1


def runLinearSVCClassifier(X_train_scale, X_val_scale, y_train, y_test):
    print(f'Training a LinearSVC Model...')
    clf = LinearSVC(
        max_iter=100000,
        tol=1e-3,
        dual=False
    )

    # Train the model using the training sets
    clf.fit(X_train_scale, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_val_scale)

    f1 = f1_score(y_test, y_pred, average='binary')

    if NEPTUNE_SEND:
        neptune.send_metric(f'LinearSVC-f1', f1)
    print(f'LinearSVC:', f1)

    if TEST_EXCLUDED:
        testModelonExcluded(clf, 'LinearSVC')

    return f1


def runRandomForestClassifier(X_train_scale, X_val_scale, y_train, y_test):
    print(f'Training a RandomForestClassifier Model...')
    clf = RandomForestClassifier(class_weight='balanced', n_estimators=1000)

    # Train the model using the training sets
    clf.fit(X_train_scale, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_val_scale)

    f1 = f1_score(y_test, y_pred, average='binary')
    if NEPTUNE_SEND:
        neptune.send_metric(f'RandomForest-f1', f1)
    print(f'RandomForest:', f1)

    if TEST_EXCLUDED:
        testModelonExcluded(clf, 'RandomForest')

    return f1


def runGradientBoostingClassifier(X_train_scale, X_val_scale, y_train, y_test):
    print(f'Training a GradientBoostingClassifier Model...')
    clf = GradientBoostingClassifier()

    # Train the model using the training sets
    clf.fit(X_train_scale, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_val_scale)

    f1 = f1_score(y_test, y_pred, average='binary')

    if NEPTUNE_SEND:
        neptune.send_metric(f'GradientBoosting-f1', f1)
    print(f'GradientBoosting:', f1)

    if TEST_EXCLUDED:
        testModelonExcluded(clf, 'GradientBoosting')

    return f1


def runKNeighborsClassifier(X_train_scale, X_val_scale, y_train, y_test, neighbors=5):
    print(f'Training a KNeighborsClassifier {neighbors} Model...')
    clf = KNeighborsClassifier(n_neighbors=neighbors)

    # Train the model using the training sets
    clf.fit(X_train_scale, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_val_scale)

    f1 = f1_score(y_test, y_pred, average='binary')

    if NEPTUNE_SEND:
        neptune.send_metric(f'KNeighborsClassifier-f1', f1)
    print(f'KNeighborsClassifier {neighbors}:', f1)

    if TEST_EXCLUDED:
        testModelonExcluded(clf, 'KNeighborsClassifier')

    return f1


def runDecisionTreeClassifier(X_train_scale, X_val_scale, y_train, y_test):
    print(f'Training a KNeighborsClassifier Model...')
    clf = DecisionTreeClassifier()

    # Train the model using the training sets
    clf.fit(X_train_scale, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_val_scale)

    f1 = f1_score(y_test, y_pred, average='binary')

    if NEPTUNE_SEND:
        neptune.send_metric(f'DecisionTreeClassifier-f1', f1)
    print(f'DecisionTreeClassifier :', f1)

    if TEST_EXCLUDED:
        testModelonExcluded(clf, 'DecisionTreeClassifier')

    return f1


def runMLPClassifier(X_train_scale, X_val_scale, y_train, y_test):
    print(f'Training a MLPClassifier Model...')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(140, 140, 140), random_state=42, max_iter=10000)

    # Train the model using the training sets
    clf.fit(X_train_scale, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_val_scale)

    f1 = f1_score(y_test, y_pred, average='binary')

    if NEPTUNE_SEND:
        neptune.send_metric(f'MLPClassifier-f1', f1)
    print(f'MLPClassifier :', f1)

    if TEST_EXCLUDED:
        testModelonExcluded(clf, 'MLPClassifier')

    return f1


def runMLPClassifierBayesSearchCV(X_train_scale, X_val_scale, y_train, y_test):
    print("Training a MLPClassifier Model...")

    clf = MLPClassifier(max_iter=10000)

    # this is our parameter grid
    param_grid = {
        'hidden_layer_sizes': Categorical[(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu', 'logistic', 'identity'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': Real(0.0001, 0.9, 'log-uniform'),
        'learning_rate': ['constant', 'adaptive', 'invscaling']}

    # set up our optimiser to find the best params in 30 searches
    opt = BayesSearchCV(
        clf,
        param_grid,
        scoring='f1',
        n_iter=30,
        random_state=1234,
        verbose=5
    )

    print(opt.total_iterations)

    opt.fit(X_train_scale, y_train)

    print('Best params achieve a test score of', opt.score(X_val_scale, y_test), ':')

    print(opt.best_params_)

    print('Best params achieve a test score of', opt.best_estimator_, ':')


def printSend(name, val):
    try:
        if (val != []):
            print(f'{name} Val f1: {np.mean(val):.3f} +- {np.std(val):.3f}')
            if NEPTUNE_SEND:
                neptune.send_metric(f'{name}-f1-Mean', np.mean(val))
                neptune.send_metric(f'{name}-f1-Std', np.std(val))
        else:
            print(f'{name} is []')
    except:
        print('Error sending values...')


def getFeatures():
    features = getTrainVecs(train_corpus)
    return features


### Used for testing on documents excluded from the training corpus, e.g. training on larger documents and testing on smaller
def testModelonExcluded(clf, name):
    global train_corpus_excluded, ytrainwiki_excluded

    X = getTrainVecs(train_corpus_excluded)
    y = ytrainwiki_excluded

    # Scale Data
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X)

    y_pred = clf.predict(X_train_scale)

    f1 = f1_score(y, y_pred, average='binary')
    if NEPTUNE_SEND:
        neptune.send_metric(f'{name}-ex-f1', f1)
    print(f'{name}-ex f1:', f1)


def train_models(kfold=0, test_size=0.2, neighbors=5):
    print("Starting model training...")

    result_list = []

    X = getFeatures()
    y = ytrainwiki

    print('Got features and data..')

    if kfold == 0:  # Run without K-fold

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if SMOTE_SAMPLING:
            sm = SMOTE(sampling_strategy='minority', random_state=7)
            # Fit the model to generate the data.
            X_train, y_train = sm.fit_sample(X_train, y_train)
        elif TomekLinks_SAMPLING:
            tl = TomekLinks(sampling_strategy='majority')
            X_train, y_train = tl.fit_sample(X_train, y_train)

        # Scale Data
        scaler = StandardScaler()
        X_train_scale = scaler.fit_transform(X_train)
        X_val_scale = scaler.transform(X_test)

        # For fine tuning the model use this method
        # runModelBayesSearchCV(X_train_scale,X_val_scale, y_train, y_test)

        lr = runLogisticRegression(X_train_scale, X_val_scale, y_train, y_test)
        sgd = runSGDClassifier(X_train_scale, X_val_scale, y_train, y_test)
        sgdh = runSGDHuberClassifier(X_train_scale, X_val_scale, y_train, y_test)
        # runSVMClassifier(X_train_scale, X_val_scale, y_train, y_test)
        gnb = runGaussianNBClassifier(X_train_scale, X_val_scale, y_train, y_test)
        lsvc = runLinearSVCClassifier(X_train_scale, X_val_scale, y_train, y_test)
        rf = runRandomForestClassifier(X_train_scale, X_val_scale, y_train, y_test)
        gb = runGradientBoostingClassifier(X_train_scale, X_val_scale, y_train, y_test)
        dt = runDecisionTreeClassifier(X_train_scale, X_val_scale, y_train, y_test)
        knn = runKNeighborsClassifier(X_train_scale, X_val_scale, y_train, y_test, neighbors)
        mlp = runMLPClassifier(X_train_scale, X_val_scale, y_train, y_test)

        result_list.append(lr)
        result_list.append(sgd)
        result_list.append(sgdh)
        result_list.append(gnb)
        result_list.append(lsvc)
        result_list.append(rf)
        result_list.append(gb)
        result_list.append(dt)
        result_list.append(knn)
        result_list.append(mlp)

    else:

        if SMOTE_SAMPLING:
            sm = SMOTE(sampling_strategy='minority', random_state=42)
            X, y = sm.fit_sample(X, y)
        elif TomekLinks_SAMPLING:
            tl = TomekLinks(sampling_strategy='majority')
            X, y = tl.fit_sample(X, y)

        cv_lr_f1, cv_lrsgd_f1, cv_svcsgd_f1, cv_svclinear_f1, cv_gnb_f1, cv_lsvc_f1, cv_rndf_f1, cv_gb_f1, cv_knn_f1 = [], [], [], [], [], [], [], [], []
        cv_mlp_f1, cv_dt_f1 = [], []

        kf = StratifiedKFold(kfold, shuffle=True, random_state=42)
        X = np.array(X)
        y = np.array(y)

        k = 0
        for train_ind, val_ind in kf.split(X, y):
            # Assign CV IDX
            X_train, y_train = X[train_ind], y[train_ind]
            X_val, y_test = X[val_ind], y[val_ind]

            # summarize train and test composition
            train_0, train_1 = len(y_train[y_train == 0]), len(y_train[y_train == 1])
            test_0, test_1 = len(y_test[y_test == 0]), len(y_test[y_test == 1])
            print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))

            # Scale Data
            scaler = StandardScaler()
            X_train_scale = scaler.fit_transform(X_train)
            X_val_scale = scaler.transform(X_val)

            # Logisitic Regression
            lr = runLogisticRegression(X_train_scale, X_val_scale, y_train, y_test)
            cv_lr_f1.append(lr)

            # Logistic Regression SGD
            sgd = runSGDClassifier(X_train_scale, X_val_scale, y_train, y_test)
            cv_lrsgd_f1.append(sgd)

            # SGD Modified Huber
            sgd_huber = runSGDHuberClassifier(X_train_scale, X_val_scale, y_train, y_test)
            cv_svcsgd_f1.append(sgd_huber)

            # # SVM
            # svm = runSVMClassifier(X_train_scale, X_val_scale, y_train, y_test)
            # cv_svclinear_f1.append(svm)
            #
            # Gaussian NB
            gnb = runGaussianNBClassifier(X_train_scale, X_val_scale, y_train, y_test)
            cv_gnb_f1.append(gnb)

            # Linear SVC
            lsvc = runLinearSVCClassifier(X_train_scale, X_val_scale, y_train, y_test)
            cv_lsvc_f1.append(lsvc)

            # Random Forest
            rndf = runRandomForestClassifier(X_train_scale, X_val_scale, y_train, y_test)
            cv_rndf_f1.append(rndf)

            # Gradient Boosting
            gb = runGradientBoostingClassifier(X_train_scale, X_val_scale, y_train, y_test)
            cv_gb_f1.append(gb)

            # Decision Trees
            dt = runDecisionTreeClassifier(X_train_scale, X_val_scale, y_train, y_test)
            cv_dt_f1.append(dt)

            # KNN
            knn = runKNeighborsClassifier(X_train_scale, X_val_scale, y_train, y_test, neighbors)
            cv_knn_f1.append(knn)

            # MLP
            mlp = runMLPClassifier(X_train_scale, X_val_scale, y_train, y_test)
            cv_mlp_f1.append(mlp)

        print('\n')
        printSend('LR', cv_lr_f1)
        printSend('LRSGD', cv_lrsgd_f1)
        printSend('LRSGDH', cv_svcsgd_f1)
        printSend('SVM', cv_svclinear_f1)
        printSend('GaussianNB', cv_gnb_f1)
        printSend('LinearSVC', cv_lsvc_f1)
        printSend('RandomForest', cv_rndf_f1)
        printSend('GradientBoosting', cv_gb_f1)
        printSend('KNN', cv_knn_f1)
        printSend('MLP', cv_mlp_f1)
        printSend('DecisionTrees', cv_dt_f1)
        return [0]

    return result_list


# Model params
kfold = 10  # 10 for 10 fold otherwise 0 for split
test_size = 0.2

ADD_LENGTH_FEATURE = True
ADD_NAME_IN_DOC = False
SORT_TOPICS = False
NEPTUNE_SEND = False

REMOVE_NUMBERS = True

REMOVE_DUPLICATE_DOCS = False
SMOTE_SAMPLING = False
TomekLinks_SAMPLING = True

MAX_DOC_WORDS = 1706800  # All
MIN_DOC_WORDS = 5  # doc must have at least this many words
SPACE = None

SCIKIT_OPTIMIZE = False
TEST_EXCLUDED = False

if SCIKIT_OPTIMIZE:
    SPACE = [skopt.space.Integer(MIN_DOC_WORDS, MAX_DOC_WORDS, name='MIN_DOC_LEN')]
else:
    SPACE = {'MIN_DOC_LEN': MIN_DOC_WORDS}

CALLS = 500
RANDOM_CALLS = 20

if SCIKIT_OPTIMIZE:
    @skopt.utils.use_named_args(SPACE)
    def objective(**params):
        mind = params['MIN_DOC_LEN']
        print(f'Loading files min doc len: {mind}...')
        loadAllConceptFiles(params)
        print('Files loaded...')

        neptune.send_metric('MIN_DOC_LEN', mind)
        model_scores = train_models(kfold=kfold)
        result = max(model_scores)
        print('Result: ' + str(result) + ' Params: ' + str(params['MIN_DOC_LEN']))

        return -1.0 * result

if __name__ == '__main__':
    start_time = time.time()

    if NEPTUNE_SEND:
        neptune.set_project('arshad115/thesis')
        experiment_name = 'supervised_classifier_concept'
        neptune.create_experiment(name=experiment_name)

    # Run if the file cleanConceptNames.json is not found to create it from the concepts dir
    saveCleanNames()

    conceptInWikiOrDisambiguation = getConceptInWikiOrDisambiguation()
    cleanConceptNames = getCleanConceptNames()

    if SCIKIT_OPTIMIZE:
        monitor = sk_utils.NeptuneMonitor()
        results = skopt.forest_minimize(objective, SPACE, n_calls=CALLS, n_random_starts=RANDOM_CALLS,
                                        callback=[monitor])
        best_auc = -1.0 * results.fun
        best_params = results.x

        print('Finished processing everything, --- %s minutes ---' % ((time.time() - start_time) / 60), flush=True)

        print('best result: ', best_auc, flush=True)
        print('best parameters: ', best_params, flush=True)

        # sk_utils.log_results(results)
        neptune.stop()
    else:
        print(f'Loading files...')
        loadAllConceptFiles(SPACE)
        print('Files loaded...')
        # for x in range(200, 270, 10):
        x = 10
        #     lda = LdaMulticore.load(os.path.join(OUTPUT_DIRECTORY, 'models', 'wiki', f'lda_{x}.gensim'))
        if NEPTUNE_SEND:
            neptune.send_metric('MIN_DOC_LEN', SPACE['MIN_DOC_LEN'])
            neptune.send_metric('lda_model_topics', x)

        model_scores = train_models(kfold=kfold)
        result = max(model_scores)
        print('Result: ' + str(result) + ' Params: ' + str(x))
