"""
This file contains all the strings (file/folder paths, error messages, etc)
that the program can use
"""


import os


"""Full paths for folders"""
sep = os.path.sep
FOLDER_ROOT = folder_root = (
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir) + sep
)

FOLDER_GENERATED_FILES = folder_generated_files = folder_root + "generated_files" + sep
folder_output = folder_generated_files + "outputs" + sep
folder_upload = folder_generated_files + "uploads" + sep
FOLDER_DATA = folder_data = folder_root + "data" + sep
FOLDER_LOGS = folder_logs = folder_root + "logs" + sep
FOLDER_CONF = folder_root + "conf" + sep

FILE_LOG_CONF = FOLDER_CONF + "logging.yml"


""" Names for folder and file(relative paths)"""

FILE_ACORNYMDB = "acronymdb.pickle"
FILE_ARTICLE_ACRONYMDB = "article_infodb.pickle"
FILE_ARTICLE_DB = "articledb.pickle"
FILE_RAW_ARTICLE_DB = "raw_articledb.pickle"
FILE_PREPROCESSED_ARTICLE_DB = "articledb.pickle"
FILE_ARTICLE_DB_SHUFFLED = "articledb_shuffled.pickle"

FILE_TFIDF_VECTORS = "vectorizer.pickle"
FILE_LABELLED_ARTICLES = "labelled_articles.pickle"
FILE_LABELLED_ACRONYMS = "labelled_acronyms.pickle"

FILE_USER_WIKIPIDIA_ANNOTATIONS = "complete_user_annotations.csv"
FILE_USER_WIKIPEDIA_ANNOTATIONS_RAW = "complete_user_annotations_raw.csv"

"""Dataset names"""
MSH_ORGIN_DATASET = "MSHCorpus"
MSH_SOA_DATASET = "MSHCorpusSOA"
REUTERS_DATASET = "Reuters"
CS_WIKIPEDIA_DATASET = "CSWikipedia"
SCIENCE_WISE_DATASET = "ScienceWISE"
FULL_WIKIPEDIA_DATASET = "FullWikipedia"
FRENCH_WIKIPEDIA_DATASET = "French_Wikipedia"
UAD_DATASET = "UAD"

SDU_AAAI_AD_DATASET = "SDU-AAAI-AD"
SDU_AAAI_AD_DEDUPE_DATASET = "SDU-AAAI-AD-dedupe"

SDU_AAAI_AI_DATASET = "SDU-AAAI-AI"

USERS_WIKIPEDIA = "UsersWikipedia"

SH_DATASET = "SH-BioC"
AB3P_DATASET = "Ab3P-BioC"
BIOADI_DATASET = "BioADI-BioC"
MEDSTRACT_DATASET = "MEDSTRACT"

DB_WITH_LINKS_SUFFIX = "_with_links"

folder_lda = folder_generated_files + "lda" + sep

folder_msh_corpus = folder_data + "MSHCorpus" + sep
folder_msh_arff = folder_msh_corpus + "arff" + sep
folder_msh_generated = folder_generated_files + "MSHCorpus" + sep

folder_msh_corpus_soa = folder_data + "MSHCorpusSOA" + sep
folder_msh_soa_generated = folder_generated_files + "MSHCorpusSOA" + sep
folder_cs_wikipedia_corpus = folder_data + "CSWikipedia" + sep

folder_cs_wikipedia_generated = folder_generated_files + "CSWikipedia" + sep


folder_reuters_articles = folder_data + "Reuters" + sep
folder_reuters_generated = folder_generated_files + "Reuters" + sep

folder_scienceWise = folder_data + "ScienceWISE" + sep
folder_scienceWise_pdfs = folder_scienceWise + "pdfs" + sep
folder_scienceWise_abstracts = folder_scienceWise + "abstracts" + sep


folder_doc2vecs = folder_generated_files + "models_context"

for folder in [
    folder_scienceWise_abstracts,
    folder_output,
    folder_data,
    folder_generated_files,
    folder_cs_wikipedia_generated,
    folder_msh_generated,
    folder_msh_soa_generated,
    folder_reuters_generated,
    folder_upload,
    folder_logs,
    folder_lda,
    folder_msh_corpus,
    folder_msh_arff,
    folder_scienceWise,
    folder_scienceWise_pdfs,
    folder_doc2vecs,
]:
    if not os.path.exists(folder):
        os.makedirs(folder)


""" Data Word Sets """
FOLDER_WORDSETS = FOLDER_DATA + "WordSets" + sep
FILE_ENGLISH_WORDS = file_english_words = FOLDER_WORDSETS + "wordsEn.txt"
FILE_JUST_ENGLISH_DICT = FOLDER_WORDSETS + "2of12inf.txt"

FILE_PREPOSITIONS = FOLDER_WORDSETS + "prepositions.txt"
FILE_DETERMINERS = FOLDER_WORDSETS + "determiners.txt"
FILE_PARTICLES = FOLDER_WORDSETS + "particles.txt"
FILE_CONJUNCTIONS = FOLDER_WORDSETS + "conjunctions.txt"


""" Pre-Trained Models """
FOLDER_PRE_TRAINED_MODELS = FOLDER_DATA + sep + "PreTrainedModels" + sep
FILE_GLOVE_EMBEDDINGS = (
    FOLDER_PRE_TRAINED_MODELS + "GloveEmbeddings" + sep + "glove.840B.300d.txt"
)
FILE_WORD2VEC_GOOGLE_NEWS = (
    FOLDER_PRE_TRAINED_MODELS + "Word2Vec" + sep + "GoogleNews-vectors-negative300.bin"
)


FILE_LUKE_PRETRAINED_MODEL = (
    FOLDER_PRE_TRAINED_MODELS + "LUKE" + sep + "luke_large_ed.tar.gz"
)

FOLDER_SCIBERT_UNCASED = (
    FOLDER_PRE_TRAINED_MODELS + sep + "scibert_scivocab_uncased" + sep
)

FOLDER_SCIBERT_CASED = FOLDER_PRE_TRAINED_MODELS + sep + "scibert_scivocab_cased" + sep

FOLDER_ALBERT = FOLDER_PRE_TRAINED_MODELS + "albert" + sep

FOLDER_MADDOG_PRE_TRAINED_MODELS = FOLDER_PRE_TRAINED_MODELS + sep + "MadDog" + sep

file_errorpage = "500.html"
file_homepage = "index.html"

file_crossvalidation_report_csv = folder_logs + "report_crossvalidation.csv"
file_report_csv = folder_logs + "report.csv"

REPORT_OUT_EXPANSION_NAME = "out-expansion"

REPORT_IN_EXPANSION_NAME = "in-expansion"

REPORT_END_TO_END_NAME = "end-to-end"

FILE_REPORT_EXTRACTION_CSV = folder_logs + "report_extraction.csv"

file_benchmark_report_pickle = folder_logs + "report_benchmark.pickle"
file_benchmark_report = folder_logs + "report_benchmark.txt"

FILE_REPORT_ENDTOEND_CSV = folder_logs + "report_endtoend.csv"

file_lda_articleIDToLDA = folder_lda + "articleIDToLDA.pickle"
file_lda_bow_corpus = folder_lda + "bow_corpus.bin"
file_lda_gensim_dictionary = folder_lda + "gensim_dictionary.bin"
file_lda_model = folder_lda + "lda_model.bin"
file_lda_word_corpus = folder_lda + "temp_word_corpus.bin"
file_lda_model_all = "lda_model_all"

file_logs = folder_logs + "log.txt"

file_msh_articleDB = folder_msh_generated + "articledb.pickle"
file_msh_articleDB_shuffled = folder_msh_generated + "articledb_shuffled.pickle"
file_msh_acronymDB = folder_msh_generated + "acronymdb.pickle"
file_msh_manual_corrections = folder_msh_corpus + "ManualCorrections.csv"
file_msh_articleIDToAcronymExpansions = (
    folder_msh_generated + "articleIDToAcronymExpansions.pickle"
)

file_reuters_articleDB = folder_reuters_generated + "articledb.pickle"
file_reuters_articleDB_shuffled = folder_reuters_generated + "articledb_shuffled.pickle"
file_reuters_acronymDB = folder_reuters_generated + "acronymdb.pickle"
file_reuters_articleIDToAcronymExpansions = (
    folder_reuters_generated + "articleIDToAcronymExpansions.pickle"
)

file_cs_wikipedia_articleDB = folder_cs_wikipedia_generated + "articledb.pickle"
file_cs_wikipedia_articleDB_shuffled = (
    folder_cs_wikipedia_generated + "articledb_shuffled.pickle"
)
file_cs_wikipedia_acronymDB = folder_cs_wikipedia_generated + "acronymdb.pickle"
file_cs_wikipedia_articleIDToAcronymExpansions = (
    folder_cs_wikipedia_generated + "articleIDToAcronymExpansions.pickle"
)


file_ScienceWise_index_train = folder_scienceWise + "sw_train_abstracts.csv"
file_ScienceWise_index_train_noabbr = (
    folder_scienceWise + "sw_train_noabbr_abstracts.csv"
)
file_ScienceWise_index_test = folder_scienceWise + "sw_test_abstracts.csv"

file_scraped_article_info = folder_data + "scraped_article_info.csv"
file_scraped_articles_list = [folder_data + "scraped_articles.csv"]
file_scraped_definitions_list = [folder_data + "scraped_definitions.csv"]

file_vectorizer = folder_generated_files + "vectorizer"
file_doc2vec = "doc2vec"
FILE_LSTM_DUAL_ENCODER = "lstm-dual-encoder"
FILE_ALBERT = "albert"


"""Miscellaneous"""
name_logger = "acronym_disambiguator"
string_unexpanded_acronym = "___EXPANSION_NOT_FOUND___"
MAX_CONFIDENCE = max_confidence = 10.0
min_confidence = -5.0

"""Error strings"""
string_error_no_results_to_show = (
    "No acronyms (between 3 and 8 letters long) were found"
)
string_error_document_parse = "The document could not be parsed.  Please try again with plaintext, or a different document."

"""Usefull French data structures"""
FR_MAPPING = {
    "é": "e",
    "è": "e",
    "à": "a",
    "’": "'",
    "ï": "i",
    "É": "E",
    "ô": "o",
    "\xa0": " ",
}
APOSTROPHE_LETTERS = [
    "A",
    "L",
    "l",
    "d",
    "D",
    "d'",
    "D'",
    "c'",
    "t'",
    "s'",
    "C'",
    "L'",
    "l'",
    "C",
    "n'",
    "N'",
]
FR_PREPOSITIONS = [
    "of",
    "sur",
    "ni",
    "ils",
    "et",
    "cet",
    "aux",
    "ces",
    "et",
    "du",
    "de",
    "au",
    "une",
    "un",
    "donc",
    "par",
    "a",
    "des",
    "tel",
    "tels",
    "se",
    "en",
    "qui",
    "que",
    "l",
    "sa",
    "les",
    "le",
    "la",
    "hui",
    "mais",
    "dont",
    "tout",
    "dans",
    "son",
    "il",
    "elle",
    "si",
    "avec",
    "cette",
    "cet",
    "ou",
    "ainsi",
    "ci",
    "lorsque",
    "cette",
    "celle",
    "ci",
    "pas",
    "La",
    "ne",
    "cependant",
    "ses",
    "the",
]
SEPARATOR_CHAR = [
    "=",
    "'",
    " ",
    ",",
    ":",
    "(",
    ")",
    "[",
    "]",
    ";",
    "«",
    "»",
    "?",
    "!",
    '"',
    "/",
    "–",
    "{",
    "}",
]
ROMAN_NUMERALS = ["I", "V", "X", "L"]
FR_VOWELS = ["a", "e", "i", "o", "u", "y", "A", "E", "I", "O", "U", "Y"]
