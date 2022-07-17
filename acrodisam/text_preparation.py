import re
import inflect
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from Logger import logging
from helper import TrainInstance

#from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.porter import PorterStemmer
#from nltk.stem.wordnet import WordNetLemmatizer

logger = logging.getLogger(__name__)

stop_words_old = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
junk_words = ["", "\n", "\r\n"]
all_words_to_remove = stop_words_old + junk_words

inflect_engine = inflect.engine()

stop_words = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

def get_expansion_without_spaces(expansion):
    # prevents preprocessings from breaking the expansion
    aux = re.sub("\W", " ", expansion).strip()
    return re.sub(r"\s+", "_", aux)

def get_singular_acronym(acronym):
    if acronym[-1] == 's':
        return acronym[:-1]
    return acronym

def acro_exp_dict_to_acro_singulars(acro_exp_dict):
    new_acro_exp_dict = {}
    for acronym, expansion in acro_exp_dict.items():
        singular_acronym = get_singular_acronym(acronym)
        new_acro_exp_dict.get(singular_acronym,[])

def sub_singulars_plurals(orig, repl, string, flags=0, orig_is_acronym=False):
    is_sub = False
    if orig_is_acronym:
        if orig[-1] == 's':
            strings_to_replace = (orig[:-1], orig)
        else:
            strings_to_replace = (orig, orig+"s")
    else:
        try:
            singular = inflect_engine.singular_noun(orig)
        except IndexError:
            logger.info("Unable to find singular of %s, probably contains an invalid character."
                        , orig)
            singular = False
        except Exception:
            logger.exception("Unexpected error when finding the singular of %s."\
                        , orig)
            singular = False
            
        if singular is False:
            #is singular
            try:
                strings_to_replace = (orig, inflect_engine.plural(orig))
            except IndexError:
                logger.info("Unable to find plural of %s, probably contains an invalid character."\
                            +" Continuing with original string only.", orig)
                strings_to_replace = (orig)
            except Exception:
                logger.exception("Unexpected error when finding the plural of %s."\
                                 +" Continuing with original string only.", orig)
                strings_to_replace = (orig)
        else:
            #is plural
            strings_to_replace = (orig, singular)

                
    new_string = string
    for search_str in strings_to_replace:
        regex_s = re.compile("\\b" + re.escape(search_str) + "\\b",flags=flags)
        #new_string = re.sub("\\b" + re.escape(s) + "\\b", repl, string, flags=flags)
        (new_string, sub_count) = regex_s.subn(repl, new_string)
        if sub_count > 0:
            is_sub = True
        
    return new_string, is_sub

def transform_text_with_exp_tokens(acronym, expansion, text):
    expansion_without_spaces = get_expansion_without_spaces(expansion)
    
    new_text, is_sub_exp = sub_singulars_plurals(expansion, expansion_without_spaces, text, flags=re.IGNORECASE)
    
    new_text, is_sub_acro = sub_singulars_plurals(acronym, expansion_without_spaces, new_text, flags=re.IGNORECASE, orig_is_acronym=True)

    return new_text, is_sub_exp or is_sub_acro, expansion_without_spaces


def transform_text_with_expansions_tokens(text, acro_exp_dict):
    expansions_without_spaces = []
    acro_exp_not_found = []
    for acronym, expansion in acro_exp_dict.items():
        if expansion is not None:
            text, success, exp_without_spaces = transform_text_with_exp_tokens(acronym, expansion, text)
            if success:
                expansions_without_spaces.append(exp_without_spaces)
            else:
                acro_exp_not_found.append((acronym, expansion))
            
    return text, expansions_without_spaces, acro_exp_not_found

def sub_expansion_tokens_by_acronym(acronym, expansion, text):
    expansion_without_spaces = get_expansion_without_spaces(expansion)
    regex_s = re.compile("\\b" + re.escape(expansion_without_spaces) + "\\b")
    (new_text, sub_count) = regex_s.subn(acronym, text)
    return new_text, sub_count

def word_tokenizer_and_transf(text, expansions_without_spaces=None, word_transf_func = lambda t:t):
    text = re.sub('\W',' ', text)
    
    tokens = word_tokenize(text)
    #stopped_tokens = [t for t in tokens if not t.lower() in stop_words]
    number_tokens = []
    for token in tokens:
        if expansions_without_spaces and token in expansions_without_spaces:
            number_tokens.append(token)
        elif token.isalnum() and not token.isdigit():
            lower_token = token.lower()
            if lower_token not in stop_words:
                number_tokens.append(word_transf_func(lower_token))
    
    return number_tokens

def text_word_tokenization(text, expansions_without_spaces=None):
    number_tokens = word_tokenizer_and_transf(text, expansions_without_spaces)
    return ' '.join(number_tokens)

def full_text_preprocessing(text, expansions_without_spaces=None):
    tokens = word_tokenizer_and_transf(text, expansions_without_spaces=expansions_without_spaces, word_transf_func=p_stemmer.stem)
    return ' '.join(tokens)

def preprocessed_text_tokenizer(text):
    return text.split(' ')

# acronymExpansions is a dict of acronym: expansion
def getTextWithExpansionTokens(text, acronymExpansions):
    for expansion in acronymExpansions.values():
        expansionWithoutSpaces = get_expansion_without_spaces(expansion)
        text = text.replace(expansionWithoutSpaces, expansion)
    return text

def preProcessArticle(acroInstance, 
                      trainArticlesDB, 
                      numSeq = None,
                      tokenizer = lambda text : tokenizePreProcessedArticleRemoveEmptyStrings(text)): # TODO consider other tokenizer
    # TODO add sp, sp.piece_to_id [CLS] <span>
        
    if isinstance(acroInstance, TrainInstance):
        term = get_expansion_without_spaces(acroInstance.expansion)

    else:
        term = acroInstance.acronym
        
    text = acroInstance.getText(trainArticlesDB)
    
    if not numSeq:
        return tokenizer(text.replace(term, ""))
    
    chuncks = text.lower().split(term.lower())
    preProcessedChuncks = [tokenizer(chunk) for chunk in chuncks]
    currTokensPerChunk = [len(chunk) for chunk in preProcessedChuncks]

    if sum(currTokensPerChunk) < numSeq:
        return [token for chunk in preProcessedChuncks for token in chunk]

    retrieveTtokensPerChunk = [0] * len(currTokensPerChunk)

    # Find how many tokens to get from each chunk
    # TODO if slow
    #n_tokens_per_chunk, remainer_n_tokens = divmod(numSeq, (len(preProcessedChuncks) - 1) * 2)
    i = 0
    while numSeq > 0:
        if i == 0 or i == len(currTokensPerChunk) - 1:
            if currTokensPerChunk[i] > 0:
                currTokensPerChunk[i] -=1
                retrieveTtokensPerChunk[i] += 1
                numSeq -= 1
        else:
            if currTokensPerChunk[i] > 1 and numSeq > 1:
                currTokensPerChunk[i] -=2
                retrieveTtokensPerChunk[i] += 2
                numSeq -= 2
            elif currTokensPerChunk[i] > 0 and numSeq > 0:
                currTokensPerChunk[i] -= 1
                retrieveTtokensPerChunk[i] += 1
                numSeq -= 1       
            
        i = (i + 1) % len(currTokensPerChunk)
    
    
    # Put chunks together
    finalTokens = []
    for i in range(len(retrieveTtokensPerChunk)):
        tokens = preProcessedChuncks[i]
        # put all
        #if retrieveTtokensPerChunk[i] < 1: 
        #    finalTokens.extend(tokens)
        
        if retrieveTtokensPerChunk[i] > 0:
            # first chunk
            if i == 0:
                n_tokens = retrieveTtokensPerChunk[i]
                finalTokens.extend(tokens[0 - n_tokens:])
            # last chunk
            elif i == len(retrieveTtokensPerChunk) - 1:
                n_tokens = retrieveTtokensPerChunk[i]
                finalTokens.extend(tokens[:n_tokens])
            # others
            elif retrieveTtokensPerChunk[i] > 0:
                n_tokens, remainer = divmod(retrieveTtokensPerChunk[i], 2)
                
                left_n_tokens = n_tokens
                if remainer > 0:
                    left_n_tokens += 1
                    
                finalTokens.extend(tokens[:left_n_tokens])
                if n_tokens > 0:
                    finalTokens.extend(tokens[0 - n_tokens:])
    
    return finalTokens

def _removePunctuations(text):
    
    #The pattern below is obtained by running: '[%s]' % re.escape(u",;:.!?-\"'&()\/`[]\u00AD<>")
    #The punctuation symbols are from string.punctuation
    #See http://stackoverflow.com/a/265995/681311 for details
    pattern_for_punctuations = u'[\\,\\;\\:\\.\\!\\?\\-\\"\\\'\\&\\(\\)\\\\\\/\\`\\[\\]\\\xad\\<\\>\\%\\+\\=]'
    cleaned_text = re.sub(pattern_for_punctuations, " ", text)

    return cleaned_text

def tokenizePreProcessedArticleRemoveEmptyStrings(text):
    return text.split()

def tokenizePreProcessedArticle(text):
    return text.split(' ')

def getCleanedWords(text, stem_words, removeNumbers):
    cleaned_text = _removePunctuations(text)
    words = word_tokenize(cleaned_text)

    result = []
    if(stem_words):
        # WordNetLemmatizer()#LancasterStemmer()#PorterStemmer()#SnowballStemmer("english")
        stemmer = LancasterStemmer()
        result = [stemmer.stem(word).lower() for word in words if word not in all_words_to_remove]
    else:
        result = [word.lower() for word in words if word.lower() not in all_words_to_remove]
    
    if(removeNumbers):
        result = [word for word in result if not word.isdigit()]
        
    return result

def toUnicode(text):
    if isinstance(text, str):
        return text.decode("utf-8", errors='ignore')
    elif isinstance(text, unicode):
        return text
    else:
        raise Exception("text is not ascii or unicode")

def toAscii(text):
    if isinstance(text, str):
        return str
    elif isinstance(text, unicode):
        return text.encode("utf-8", "backslashreplace")
    else:
        raise Exception("text is not ascii or unicode")