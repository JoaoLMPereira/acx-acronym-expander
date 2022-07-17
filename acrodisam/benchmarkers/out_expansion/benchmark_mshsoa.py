import sys

import pickle

from benchmarkers.out_expansion.benchmark import Benchmarker
from string_constants import MSH_SOA_DATASET
from helper import getDatasetGeneratedFilesPath

from Logger import logging

logger = logging.getLogger(__name__)


class Benchmarker_MSHSOA(Benchmarker):
    
    dataset_name = MSH_SOA_DATASET
    # sets defaults

    tfidfArgs = ['0-0.50-5', '3-3']
    ldaArgs = ['100', 'log(nub_distinct_words)+1']

    doc2vecArgs= ['100', 'CBOW', '25', '2']
    
    #ngramsContextVectorsExpArgs = ['0','0.25','1']
    ngramsContextVectorsExpArgs = ['0','1','1']

    def __init__(self, args, is_cv = False):
        experiment_name = MSH_SOA_DATASET
        if is_cv:
            experiment_name += "_CV"
        super().__init__(experiment_name=experiment_name, train_dataset_name=MSH_SOA_DATASET, args=args)

if __name__ == "__main__":
    argv = sys.argv[1:]
    logger.info("Args: %s", ' '.join(argv))
    if(argv[0] == "new"):
        argv = argv[1:]
        newSuffix = "_new"
    else:
        newSuffix = ""
    
    if(argv[0] == "CV"):
        argv = argv[1:]
        isCV = True
    else:
        isCV = False
    benchmarker = Benchmarker_MSHSOA(argv, isCV)
    generatedFilesFolder = getDatasetGeneratedFilesPath(MSH_SOA_DATASET)

    if isCV:
        foldsNum = 5
        foldsFilePath = generatedFilesFolder + str(foldsNum) + '-cross-validation'+newSuffix+'.pickle'
        folds = pickle.load(
                open(foldsFilePath, "rb"))
        #Adds index to folds list
        indexedFolds = [(fold[0], fold[1], idx) for idx, fold in enumerate(folds)]
    else:
        trainArticles = pickle.load(open(generatedFilesFolder + 'train_articles'+newSuffix+'.pickle', "rb"))
        testArticles = pickle.load(open(generatedFilesFolder + 'test_articles'+newSuffix+'.pickle', "rb"))
        indexedFolds = [(testArticles, trainArticles)]

    if isCV and (len(argv) > 0 and argv[0] == 'NGramsContextVector' 
             or len(argv) > 1 and argv[1] == 'NGramsContextVector'):
        num_processes = 3
    else:
        num_processes = len(indexedFolds)
    
    benchmarker.run(indexedFolds, num_processes=num_processes)
    
    
