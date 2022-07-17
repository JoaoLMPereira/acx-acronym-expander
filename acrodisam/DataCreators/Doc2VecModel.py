import os
import tempfile
import shutil
import logging
import pickle
from DataCreators import ArticleDB, ArticleAcronymDB
from string_constants import file_doc2vec, MSH_SOA_DATASET
from gensim.models.doc2vec import Doc2Vec, TaggedDocument, TaggedLineDocument, Doctag
from nltk.tokenize import word_tokenize
from helper import getDatasetGeneratedFilesPath, display_top, logOSStatus
import text_preparation


logger = logging.getLogger(__name__)

"""
def getTextCorpus(articleDB):
    text_corpus = {}
    for article_id, text in articleDB.items():
        text_corpus[article_id] = text
    return text_corpus
"""

def getDoc2VecModel(corpus, epochs=5, dm=1, vector_size=100, window=5):
    doc2vecModel = Doc2Vec(epochs= epochs, dm=dm, vector_size=vector_size, window=window, workers=3)
    doc2vecModel.build_vocab(corpus)

    doc2vecModel.train(corpus,
                    total_examples=doc2vecModel.corpus_count,
                    epochs=epochs)
    
    return doc2vecModel

def preProcessText(text):
    return text_preparation.tokenizePreProcessedArticle(text.lower())


def documentExpansionTagGenerator(articleDB, articleAcronymDB):
    for i, _d in articleDB.items():
        acronymExpansions = articleAcronymDB.get(i)
        if acronymExpansions:
            tags = [exp for _, exp in acronymExpansions.items()]# + [str(i)]
            yield TaggedDocument(words=preProcessText(_d), tags=tags)

def pickleLoader(pklFile):
    try:
        while True:
            yield pickle.load(pklFile)
    except EOFError:
        pass

def trainDoc2VecModel(articleDB, articleAcronymDB=None, expansionAsTags=False,
                      epochs=5, dm=1, vector_size=100, window=5,
                                datasetName=None,
                                fold="",
                                saveAndLoad=False,
                                persistentArticles=None,
                                executionTimeObserver=None):
    
   
    if saveAndLoad:
        generatedFilesFolder = getDatasetGeneratedFilesPath(datasetName)
        if datasetName.endswith("_confidences"):
            datasetName = datasetName.replace("_confidences","")
        varToName = "_".join([str(s) for s in [datasetName,fold,epochs,dm,vector_size,window]])
        if expansionAsTags:
            varToName += "_EXP"
        doc2VecLocation = generatedFilesFolder + file_doc2vec + "_" + varToName + ".pickle"
        tagsMapLocation = generatedFilesFolder + "doc2vectagsmap_" + varToName + ".pickle"
            
        if os.path.isfile(doc2VecLocation):
            tagsMap = None       
            if os.path.isfile(tagsMapLocation) and not expansionAsTags:
                with open(tagsMapLocation, "rb") as f:
                    tagsMap = pickle.load(f)
                        
            return Doc2Vec.load(doc2VecLocation, mmap='r'), tagsMap
        logger.info("Doc2Vec not loaded, file not found: %s", doc2VecLocation)
    tagsMap = None
    import gc
    import tracemalloc
    #text_corpus = getTextCorpus(articleDB)
    text_corpus = articleDB
    if persistentArticles == None:
        logger.info("Doc2Vec without persistent preprocessed articles")
        if expansionAsTags:
            tagged_data = list(documentExpansionTagGenerator(articleDB, articleAcronymDB))
            #for i, _d in text_corpus.items():
            #    acronymExpansions = articleAcronymDB.get(i)
            #    if acronymExpansions:
            #        tags = [exp for _, exp in articleAcronymDB.get(i)]
            #        tagged_data.append(TaggedDocument(words=preProcessText(_d), tags=tags))
        else:
            #tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in text_corpus.items()]
            tagged_data = [TaggedDocument(words=preProcessText(_d), tags=[str(i)]) for i, _d in text_corpus.items()]
        if executionTimeObserver:
            executionTimeObserver.start()
        doc2vecModel = getDoc2VecModel(tagged_data, epochs=epochs, dm=dm, vector_size=vector_size, window=window)
        if executionTimeObserver:
            executionTimeObserver.stop()

    else:
        logger.info("Doc2Vec with persistent preprocessed articles")
        if expansionAsTags:
            documents_iterator = documentExpansionTagGenerator(articleDB, articleAcronymDB)
            
            tmpDirName = tempfile.mkdtemp()
            try:
                corpusFileName = tmpDirName + "/corpus.pickle"
                
                #gensim.utils.save_as_line_sentence(corpus, tmpDirName + "corpus")
                with open(corpusFileName, "wb") as f:
                    for taggedDocument in documents_iterator:
                        pickle.dump(taggedDocument, f, pickle.HIGHEST_PROTOCOL)

                with open(corpusFileName, "rb") as f:
                    documents_file_iterator = pickleLoader(f)
                    # TODO make generator from file
                    if executionTimeObserver:
                        executionTimeObserver.start()
                    doc2vecModel = getDoc2VecModel(documents_file_iterator, epochs=epochs, dm=dm, vector_size=vector_size, window=window)
                    
                    if executionTimeObserver:
                        executionTimeObserver.stop()
            finally:
                shutil.rmtree(tmpDirName)
                
        else:
            tmpDirName = tempfile.mkdtemp()
            try:
                corpusFileName = tmpDirName + "/corpus.txt"
                tagsMap = {}
                #gensim.utils.save_as_line_sentence(corpus, tmpDirName + "corpus")
                with open(corpusFileName, "w") as f:
                    i = 0
                    for tag, _d in text_corpus.items():
                        #f.write(' '.join(preProcessText(_d))+'\n')
                        processedText = ' '.join(preProcessText(_d)) # TODO tokenizer alterado do NLTK
                        #processedText = ' '.join(word_tokenize(_d.lower()))
                        f.write(processedText+'\n')
                        tagsMap[str(tag)] = i
                        i = i + 1
                    logger.critical("i = " + str(i))    
                logger.critical("Before Doc2Vec(corpus_file=corpusFileName, epochs=epochs, dm=dm, vector_size=vector_size, window=window)")
                if tracemalloc.is_tracing():
                    gc.collect()
                    snapshot3 = tracemalloc.take_snapshot()
                    display_top(snapshot3, limit = 20)
                logOSStatus()
                if executionTimeObserver:
                    executionTimeObserver.start()
                doc2vecModel = Doc2Vec(corpus_file=corpusFileName, epochs=epochs, dm=dm, vector_size=vector_size, window=window, workers=8)
                if executionTimeObserver:
                    executionTimeObserver.stop()
                #logger.critical("len(doc2vecModel) = " + len(doc2vecModel))
            finally:
                shutil.rmtree(tmpDirName)
                #os.rmdir(tmpDirName)
            logger.critical("After Doc2Vec(corpus_file=corpusFileName, epochs=epochs, dm=dm, vector_size=vector_size, window=window)")
            if tracemalloc.is_tracing():
                gc.collect()
                snapshot3 = tracemalloc.take_snapshot()
                display_top(snapshot3, limit = 20)
            logOSStatus()
            """
            TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)])
            with tempfile.TemporaryFile() as f:
                for i, _d in text_corpus.items():
                    f.writeline(' '.join(word_tokenize(_d.lower())))
                
                tagged_data = TaggedLineDocument(f)
                doc2vecModel = getDoc2VecModel(tagged_data, epochs=epochs, dm=dm, vector_size=vector_size, window=window)
            """

    doc2vecModel.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    logger.critical("After doc2vecModel.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)")    
    if tracemalloc.is_tracing():
        gc.collect()
        snapshot3 = tracemalloc.take_snapshot()
        display_top(snapshot3, limit = 20)
    logOSStatus()
    
    
    if saveAndLoad:
        doc2vecModel.save(doc2VecLocation)
        if tagsMap and not expansionAsTags:
            with open(tagsMapLocation, "wb") as f:
                pickle.dump(tagsMap, f, protocol=1)
    
    return doc2vecModel, tagsMap

def main():
    datasetName = MSH_SOA_DATASET
    articleDB = ArticleDB.getArticleDB(datasetName)
    articleAcronymDB = ArticleAcronymDB.getArticleAcronymDB(datasetName)

    expansionAsTags = True
    persistentArticles = None
    
    doc2vecModel, tagsMap = trainDoc2VecModel(articleDB, articleAcronymDB, expansionAsTags,datasetName=datasetName,persistentArticles = persistentArticles)

if __name__ == "__main__":
    main()
