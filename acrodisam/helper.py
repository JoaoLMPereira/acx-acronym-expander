import collections
from difflib import SequenceMatcher
from itertools import zip_longest
import itertools
import json
import linecache
import logging
import os
import random
import sys
import time
import tracemalloc

from collections import OrderedDict


from configargparse import YAMLConfigFileParser, ConfigFileParserException
import configargparse
from nltk.metrics.distance import edit_distance

from string_constants import (
    FOLDER_DATA,
    FOLDER_GENERATED_FILES,
    FILE_ACORNYMDB,
    FILE_ARTICLE_ACRONYMDB,
    FILE_ARTICLE_DB,
    FILE_ARTICLE_DB_SHUFFLED,
    sep,
    FR_MAPPING,
    FILE_LABELLED_ARTICLES,
    FILE_LABELLED_ACRONYMS,
    FOLDER_CONF,
    FILE_RAW_ARTICLE_DB,
    FILE_PREPROCESSED_ARTICLE_DB,
)
from json.decoder import JSONDecodeError


class YAMLConfigFileParserCustom(YAMLConfigFileParser):
    def parse(self, stream):
        """Parses the keys and values from a config file."""
        yaml = self._load_yaml()

        try:
            parsed_obj = yaml.safe_load(stream)
        except Exception as ex:
            raise ConfigFileParserException(
                "Couldn't parse config file: %s" % ex
            ) from ex

        if not isinstance(parsed_obj, dict):
            raise ConfigFileParserException(
                "The config file doesn't appear to "
                "contain 'key: value' pairs (aka. a YAML mapping). "
                "yaml.load('%s') returned type '%s' instead of 'dict'."
                % (getattr(stream, "name", "stream"), type(parsed_obj).__name__)
            )

        result = OrderedDict()
        for key, value in parsed_obj.items():
            if isinstance(value, list):
                result[key] = json.dumps(value)
            elif isinstance(value, dict):
                result[key] = json.dumps(value)
            else:
                result[key] = str(value)

        return result


def get_conf_path():
    return FOLDER_CONF + "".join(sys.argv[0].split("/")[-1].split(".")[:-1]) + ".yml"


def _json_arg_parse(arg_value):
    try:
        args = json.loads(arg_value)
    except JSONDecodeError as e:
        args = [arg_value]
    return args


# Code to get file and folder paths
def create_configargparser(
    crossvalidation=False,
    in_expander=False,
    external_data=False,
    out_expander=False,
    links=False,
    save_and_load=False,
    report_confidences=False,
    results_db_config=False,
):
    parser = configargparse.ArgParser(
        default_config_files=[FOLDER_CONF + "configfile.yml", get_conf_path()],
        config_file_parser_class=YAMLConfigFileParserCustom,
    )

    parser.add(
        "-c",
        "--my-config",
        required=False,
        is_config_file=True,
        help="config yaml file path",
    )

    if report_confidences:
        parser.add_argument(
            "--report_confidences",
            "-rc",
            action="store_true",
            help="saves confidences for each expansion into a file, this is later used by assembling method",
        )

    if results_db_config:
        parser.add_argument(
            "--results_database_configuration",
            "-db_conf",
            default=None,
            type=json.loads,
            help="configuration to access the database where the experiment results will be stored",
        )

    if crossvalidation:
        parser.add_argument(
            "--crossvalidation",
            "-cv",
            action="store_true",
            help="runs cross-validation over the Train Data",
        )

    if links:
        parser.add_argument(
            "--follow-links",
            "-fl",
            action="store_true",
            help="acronym expander may follow links to expand acronyms",
        )

    if save_and_load:
        parser.add_argument(
            "--save-and-load",
            "-sl",
            action="store_true",
            help="""if supported loads the model if it was saved before,
             otherwise creates the model and saves it into disk""",
        )
        
    if external_data:
        parser.add_argument(
            "--external-data",
            "-ed",
            action="store_true",
            help="""Uses external data for training""",
        )

    if in_expander:
        parser.add_argument(
            "--in_expander",
            default="schwartz_hearst_original",
            # choices=get_available_out_expanders(),
            help="in-expander name",
        )
        parser.add_argument(
            "--in_expander_args",
            default=None,
            type=_json_arg_parse,
            help="arguments for the in-expander if more than one format with Json",
        )

    if out_expander:
        from out_expanders import (
            get_available_out_expanders,
        )  # pylint: disable=import-outside-toplevel

        parser.add_argument(
            "--out_expander",
            default="classic_context_vector",
            choices=get_available_out_expanders(),
            help="out-expander name",
        )
        parser.add_argument(
            "--out_expander_args",
            default=None,
            type=_json_arg_parse,
            help="arguments for the out-expander if more than one, format with Json",
        )
        
    return parser


def mkdirIfNotExists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def getDatasetPath(datasetName):
    path = FOLDER_DATA + datasetName + sep
    mkdirIfNotExists(path)
    return path


def getDatasetGeneratedFilesPath(datasetName):
    if datasetName.endswith("_confidences"):
        datasetName = datasetName.replace("_confidences","")
    path = FOLDER_GENERATED_FILES + datasetName + sep
    mkdirIfNotExists(path)
    return path


def get_acronym_db_path(dataset_name):
    return getDatasetGeneratedFilesPath(dataset_name) + FILE_ACORNYMDB


def getArticleAcronymDBPath(datasetName):
    return getDatasetGeneratedFilesPath(datasetName) + FILE_ARTICLE_ACRONYMDB


def getArticleDBPath(datasetName):
    return getDatasetGeneratedFilesPath(datasetName) + FILE_ARTICLE_DB


def get_raw_article_db_path(datasetName):
    return getDatasetGeneratedFilesPath(datasetName) + FILE_RAW_ARTICLE_DB


def get_preprocessed_article_db_path(datasetName):
    return getDatasetGeneratedFilesPath(datasetName) + FILE_PREPROCESSED_ARTICLE_DB


def getArticleDBShuffledPath(datasetName):
    return getDatasetGeneratedFilesPath(datasetName) + FILE_ARTICLE_DB_SHUFFLED


def get_labelled_articles_path(dataset_name):
    return getDatasetGeneratedFilesPath(dataset_name) + FILE_LABELLED_ARTICLES


def get_labelled_acronyms_path(dataset_name):
    return getDatasetGeneratedFilesPath(dataset_name) + FILE_LABELLED_ACRONYMS


def grouped_longest(iterable, elements_num):
    return zip_longest(*[iter(iterable)] * elements_num)


def zip_with_scalar(l, o):
    return zip(l, itertools.repeat(o))


# Code to generate folds
def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
    Copied from https://docs.python.org/2/library/itertools.html
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def getPartitions(articleDb, numRounds):
    partitions = []

    articleDbItems = list(articleDb.items())
    random.Random(1337).shuffle(articleDbItems)

    partitionSize = int(len(articleDbItems) / numRounds)
    if (len(articleDbItems) % numRounds) != 0:
        partitionSize += 1

    for ids in grouper(articleDbItems, partitionSize, fillvalue=None):
        partitions.append(list(ids))
    # articleDb.clear()
    return partitions


def extend_dict_of_lists(base_dict, ext_dict):
    for key, value in ext_dict.items():
        newDictValue = base_dict.setdefault(key, [])
        newDictValue.extend(value)
        # when using SQLite we have to make sure the value is assigned
        base_dict[key] = newDictValue


def _flatten_to_str_list_aux(l):
    items = []
    for v in l:
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_to_str_list(v.values()))
        elif isinstance(v, str):
            items.append(v)
        elif isinstance(v, collections.Iterable):
            items.extend(flatten_to_str_list(v))
        else:
            items.append(str(v))
    return list(items)


def flatten_to_str_list(v):
    if isinstance(v, collections.MutableMapping):
        return _flatten_to_str_list_aux(v.values())
    if isinstance(v, str):
        return v
    if isinstance(v, collections.Iterable):
        return _flatten_to_str_list_aux(v)
    return str(v)


def get_lang_dict(lang):
    vocab = []
    if lang == "FR":
        vocab_path = FOLDER_DATA + "FrenchData/french_words.txt"
        if os.path.isfile(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as words:
                vocab = [word.replace("\n", "") for word in words]
    else:
        raise Exception("This dict is only for French. Please specify FR as param.")
    return vocab


def verifyTrainSet(self, articleDB, acronymDB, testArticleIDs):
    for articleId in articleDB:
        if articleId in testArticleIDs:
            return False
    for acronym in acronymDB:
        for ignored_field1, articleId, ignored_field2 in acronymDB[acronym]:
            if articleId in testArticleIDs:
                return False
    return True


def __getRealignedAcronymDb(self, articleIDsToRemove, acronymDb):
    #    acronymDb = AcronymDB.load(path=self.acronymDBPath)

    for acronym in acronymDb.keys():
        validEntries = []
        for entry in acronymDb[acronym]:
            if entry[1] not in articleIDsToRemove:
                validEntries.append(entry)
        acronymDb[acronym] = validEntries

    return acronymDb


def getTrainTestData(articlesIds, trainFraction=0.80):
    articlesIds = list(articlesIds)
    trainSize = int(len(articlesIds) * trainFraction)
    trainIds = random.Random(1337).sample(articlesIds, trainSize)

    testIds = [tmpId for tmpId in articlesIds if tmpId not in trainIds]
    return trainIds, testIds


def getCrossValidationFolds(articlesIds, foldsNum):
    articlesIds = list(articlesIds)
    folds = []

    random.Random(1337).shuffle(articlesIds)

    partitionSize = int(len(articlesIds) / foldsNum)
    if (len(articlesIds) % foldsNum) != 0:
        partitionSize += 1

    for ids in grouper(articlesIds, partitionSize, fillvalue=None):
        testIds = list(ids)
        # Train ids are the remaining
        trainIds = [tmpId for tmpId in articlesIds if tmpId not in testIds]
        folds.append((testIds, trainIds))
    return folds


def sparse_memory_usage(mat):
    try:
        return mat.data.nbytes + mat.row.nbytes + mat.col.nbytes / 1000000.0
    except AttributeError:
        return -1


def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def removeSubStringAndGetOccurancesFromText(text, substring):
    idx_list = []
    finalText = ""
    for t in text.split(substring):
        finalText = finalText + t
        idx_list.append(len(finalText))

    return finalText, idx_list


def mergeDictsOfLists(d1, d2):
    newDict = d1.copy()
    for k, l in d2.items():
        newDict.setdefault(k, []).extend(l)

    return newDict


def getAcronymChoices(acronym, acronymDB):
    """
    takes in an acronym, returns the choices in the following form
    returns:
    X_train (list): of helper.ExpansionChoice
    y_labels (list): of integer labels associated with each entry in X_train
    labelToExpansion (dict): to convert label number to expansion
    """
    # get matches from acronymDB
    matches = []
    if acronym in acronymDB:
        matches += acronymDB[acronym]
    if acronym[-1] == "s" and acronym[:-1] in acronymDB:
        matches += acronymDB[acronym[:-1]]

    # create training data
    X_train, y_train = [], []
    for definition, articleID in matches:
        # text = self.articleDB[str(articleID)]
        X_train.append(
            TrainInstance(article_id=articleID, acronym=acronym, expansion=definition)
        )
        y_train.append(definition)

    # create y labels to group similar acronyms
    y_labels, labelToExpansion = _processAcronymChoices(y_train)

    return X_train, y_labels, labelToExpansion


def _processAcronymChoices(acronym_expansions):
    """
    input: list(acronym expansion strings)
    returns:
    y_labels (list): of integer labels assigned to acronym expansions
    labelToExpansion (dict): to convert label number to acronym expansion
    """
    y_labels = []
    labelToExpansion = {}

    if len(acronym_expansions) == 0:
        return y_labels, labelToExpansion

    y_labels = [index for index in range(len(acronym_expansions))]
    labelToExpansion[0] = acronym_expansions[0]

    for indexAhead in range(1, len(acronym_expansions)):
        new_expansion = acronym_expansions[indexAhead]
        newIsUnique = True

        # check if new_expansion is same as a previous expansion
        # if same assign previous label and move on
        for label, expansion in labelToExpansion.items():
            if AcronymExpansion.areExpansionsSimilar(expansion, new_expansion):
                newIsUnique = False
                y_labels[indexAhead] = label
                break
        # if label is new indeed, then give it a label ID (integer) and
        # make an entry in the labelToExpansion dictionary
        if newIsUnique:
            new_class_label = len(labelToExpansion)
            labelToExpansion[new_class_label] = new_expansion
            y_labels[indexAhead] = new_class_label

    return y_labels, labelToExpansion


def groupby_indexes_unsorted(seq, key=lambda x: x):
    indexes = collections.defaultdict(list)
    for i, elem in enumerate(seq):
        indexes[key(elem)].append(i)
    return indexes.items()


def format_text(text):
    """Remove french specific character.
    Args:
        text (str): the text to format
    Returns:
        new_text (str): thee formated text
    """
    new_text = ""
    for letter in text:
        new_text = new_text + FR_MAPPING.get(letter, letter)
    new_text = new_text.replace('"', "")
    return new_text


def get_args_to_pass(in_args, args=None, kwargs=None):
    if args is None:
        args = []
    else:
        args = list(args)

    if kwargs is None:
        kwargs = {}

    if isinstance(in_args, list):
        return in_args + args, kwargs
    elif isinstance(in_args, dict):
        return args, {**kwargs, **in_args}
    # single value, add to args list
    elif in_args is not None:
        return [in_args] + args, kwargs

    return args, kwargs


class AcronymExpansion:
    """
    Class containing an acronym's expansion
    """

    def __init__(self, expansion, confidence, options):
        """
        expansion: str of expansion
        expander: str (Enum) of expander
        confidence: float value of confidence in this expansion (ideally within [0,1])
        """
        self.expansion = expansion
        # self.expander = expander
        self.confidence = confidence
        self.options = options

    def __str__(self):
        # return self.expander + "," + self.expansion
        return self.expansion

    @staticmethod
    def areDistinctChoices(choices):
        """
        Tell whether the choices are distinct
        """
        count = len(choices)
        if count <= 1:
            return False
        else:
            res1 = choices[0].expansion.strip().lower().replace("-", " ")
            res1 = " ".join([w[:4] for w in res1.split()])
            res2 = choices[-1].expansion.strip().lower().replace("-", " ")
            res2 = " ".join([w[:4] for w in res2.split()])
            if res1 != res2:
                return True
            return False

    @staticmethod
    def startsSameWay(expansion_1, expansion_2):
        expansion_1 = expansion_1.strip().lower().replace("-", " ")
        expansion_2 = " ".join([word[:4] for word in expansion_2.split()])
        expansion_1 = " ".join([word[:4] for word in expansion_1.split()])
        if expansion_2 == expansion_1:
            return True
        #    ed = distance.edit_distance(expansion_2, expansion_1)
        #    if ed < 3:
        #        return True
        return False

    @staticmethod
    def areExpansionsSimilar(expansion_1, expansion_2):
        expansion_1 = expansion_1.lower().replace(u"-", u" ")
        expansion_2 = expansion_2.lower().replace(u"-", u" ")
        # numActualWords = len(expansion_1)
        # numPredictedWords = len(expansion_2)

        if (
            expansion_1 == expansion_2
            or AcronymExpansion.startsSameWay(expansion_1, expansion_2)
            or edit_distance(expansion_1, expansion_2) <= 2
        ):  # max(numActualWords, numPredictedWords)):
            return True

        return False

    @staticmethod
    def areExpansionsSimilarCompetitors(expansion_1, expansion_2):
        expansion_1 = expansion_1.lower().strip()
        expansion_2 = expansion_2.lower().strip()
        if (
            expansion_1 == expansion_2
            or SequenceMatcher(None, expansion_1, expansion_2).ratio() > 0.80
        ):
            return True
        return False


class AcroInstance:
    def __init__(self, acronym, article_id=None):
        self.article_id = article_id
        self.acronym = acronym


class TrainInstance(AcroInstance):
    def __init__(self, article_id, acronym, expansion):
        super().__init__(acronym, article_id)
        self.acronym = acronym
        self.expansion = expansion
        # TODO soft link article_text
        self.article_text_per_db = {}

    def getText(self, articlesDB=None):
        article_db_id = id(articlesDB)
        text = self.article_text_per_db.get(article_db_id)
        if text is None:
            text = articlesDB[self.article_id]
            self.article_text_per_db[article_db_id] = text

        return text


class TestInstance(AcroInstance):
    def __init__(self, article_id, article_text, acronym):
        super().__init__(acronym, article_id)
        self.article_text = article_text

    def getText(self, articlesDB=None):
        return self.article_text


class LDAStructure:

    #    def __init__(self, ldaModel, dictionary, articleIDToLDADict, articleDBused, stem_words, numPasses, removeNumbers):
    def __init__(self, ldaModel, dictionary, articleIDToLDADict):
        self.ldaModel = ldaModel
        self.dictionary = dictionary
        self.articleIDToLDADict = articleIDToLDADict


#        self.articleDBused = articleDBused
#        self.stem_words = stem_words
#        self.numPasses = numPasses
#        self.removeNumbers = removeNumbers


class ExecutionTimeObserver:
    def __init__(self, startTime=0, stopTime=0):
        self._startTime = startTime
        self._stopTime = stopTime

    def start(self):
        self._startTime += time.time()

    def stop(self):
        self._stopTime += time.time()

    def getTime(self):
        return self._stopTime - self._startTime

    def add(self, other):
        self._startTime += other._startTime
        self._stopTime += other._stopTime

    def __add__(self, other):
        totalStartTime = self._startTime + other._startTime
        totalStopTime = self._stopTime + other._stopTime
        return ExecutionTimeObserver(startTime=totalStartTime, stopTime=totalStopTime)

    def __str__(self):
        return str(self.getTime())


def display_top(snapshot, key_type="lineno", limit=10, previousSnapshot=None):
    logger = logging.getLogger(__name__ + ".display_top")
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    if previousSnapshot:
        top_stats = snapshot.compare_to(previousSnapshot, key_type)
    else:
        top_stats = snapshot.statistics(key_type)

    logger.critical("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        logger.critical(
            "#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            logger.critical("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        logger.critical("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    logger.critical("Total allocated size: %.1f KiB" % (total / 1024))


def logOSStatus():
    return None
    """
    logger = logging.getLogger(__name__ + ".logOSStatus")
    logger.critical("")
    logger.critical("OS status")
    p = psutil.Process()
    logger.critical("Memory: " + str(p.memory_full_info()))
    logger.critical("Opened files: " +  str(p.open_files()))
    """
