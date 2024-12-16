from typing import List, Tuple
import regex as re
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


def worldLevelGoldTag(sentence_pair:List[str], label:str) -> str:
    if label == 'Entailment':
        tag = 'Other'
    else:
        words1 = sentence_pair[0].lower().split()
        words2 = sentence_pair[1].lower().split()
        if words1.sort() == words2.sort():
            tag = 'WordSwap'
        else:
            unionSet = set(words1).union(set(words2))
            if len(words2) == len(words1) and (len(unionSet) - len(words1) <= 1
                 or len(unionSet) - len(words2) <= 1):
                diff1, diff2, index = findDiff(words1, words2)
                if diff1 == findRegex(sentence_pair[0]) and diff2 == findRegex(sentence_pair[1]):
                    # regex match quantifiers
                    tag = 'QuantSub'
                else:
                    # If both are nouns or pronouns: Obj Sub
                    tag_list1, tag_list2 = getPos(sentence_pair[0], sentence_pair[1], diff1, diff2)
                    tag1 = tag_list1[index]
                    tag2 = tag_list2[index]
                    if tag1 in ['NN', 'PRP', 'JJ', 'RB', 'VB'] and tag2 in ['NN', 'PRP', 'JJ', 'RB', 'VB']:
                        if tag1 in ['NN', 'PRP'] and tag2 in ['NN', 'PRP']:
                            tag = 'ObjSub'
                        elif tag1 in ['JJ', 'RB'] and tag2 in ['JJ', 'RB']:
                            tag = 'NegaExp'
                        elif tag1 == 'VB' and tag2 == 'VB':
                            tag = 'ActSub'
                        else:
                            tag = 'Other'
                    else:
                        tag = 'Other'
            else:
                wordset1 = set(words1)
                wordset2 = set(words2)
                if wordset1.issubset(wordset2):
                    if len(words2) - len(words1) <= 2 and containNegation(wordset2 - wordset1):
                        tag = 'NegaExp'
                    else:
                        tag = 'WordDel'
                else:
                    tag = 'Other'
    return tag



def findDiff(wordlist1: List[str], wordlist2: List[str]) -> Tuple[str, str, int]:
    for i in range(len(wordlist1)):
        if wordlist1[i] != wordlist2[i]:
            return wordlist1[i], wordlist2[i], i
    return '', '', -1


def findRegex(sentence: str) -> List[str]:
    quant = re.findall(r'-?\d+(?:\.\d=)?', sentence)
    return quant


def getPos(sentence1: str, sentence2: str, diff1: None, diff2: None)-> Tuple[List[str], List[str]]:
    # judge the components of the word in the original sentence
    # focus on 5 categories nouns, pronouns, adjectives, adverbs and verbs.
    tag_dict1 = [t[1] for t in pos_tag(word_tokenize(sentence1))]
    tag_dict2 = [t[1] for t in pos_tag(word_tokenize(sentence2))]
    return tag_dict1, tag_dict2


def containNegation(words: set) -> bool:
    # negation, e.g. not, do not, etc.
    negation_words = ["no", "not", "never", "none", "nobody", "nothing", "neither", "nowhere", "can't", "won't",
                      "don't"]
    # more words or phase ?
    return bool(words.intersection(negation_words))


def sentenceLevelGoldTag(sentence_pair: List[str], back_translation: bool = False) -> str:
    # Use heuristics to identify sentence-level relations such as back translation or inference generation
    # close in length, keywords in the sentences are basically the same
    # modifiers and its synonyms can be found in sentences
    if back_translation:
        len_diff = abs(len(sentence_pair[0]) - len(sentence_pair[1]))
        common_words = set(sentence_pair[0].split()).intersection(set(sentence_pair[1].split()))
        if len_diff <= 2 and len(common_words) / min(len(sentence_pair[0].split()),
                                                     len(sentence_pair[1].split())) > 0.7:
            return 'BackTrans'

    # Placeholder for identifying inference generation
    if isInferenceGenerated(sentence_pair):
        return 'InferGen'
    return 'Other'


def isInferenceGenerated(sentence_pair: List[str]) -> bool:
    return True
