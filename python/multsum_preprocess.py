import re

REGEX_SPACE         = " +"

def preprocess(sentences_lists, anaphora_resolution_simple=False):
  if anaphora_resolution_simple:
    list_of_lists_to_return = list()
    try:
      for sentence_list in sentences_lists:
        list_to_return = list()
        previous_person = None
        previous_person_plural = None
        for sentence in sentence_list:
          words = re.split(REGEX_SPACE, sentence)
          import nltk
          tagged = nltk.pos_tag(words)
          tags = [item[1] for item in tagged]
          if len(tagged) != len(words):
            print "length of tagged (%d) differs from number of words in sentence (%d)"%(len(tagged) != len(words_i))
            exit()
          processed_sentence = ""
          for i in xrange(0, len(words)):
            #if (tags[i] == "PRP" or tags[i] == "PRP$") and words[i] != "it":
            if is_pronoun_to_replace(words[i]):
              possessive_suffix = ""
              if is_possessive_pronoun(words[i]):
              #if tags[i][-1] == "$":
                # English bias:
                possessive_suffix = "'s"
              punctuation = ""
              last_char = words[i][-1]
              word_without_punctuation = words[i]
              if is_punctuation(last_char):
                punctuation = last_char
                word_without_punctuation = words[i][:-1]
              if is_plural_pronoun(words[i]) and previous_person_plural is not None:
                processed_sentence = processed_sentence+" "+word_without_punctuation+" ("+previous_person_plural+possessive_suffix+")"+punctuation
              elif previous_person is not None:
                processed_sentence = processed_sentence+" "+word_without_punctuation+" ("+previous_person+possessive_suffix+")"+punctuation
            else:
               if tags[i] == "NNP":
                 previous_person = words[i].strip(".?!,'\"\n ")
               elif tags[i] == "NNPS":
                 previous_person_plural = words[i].strip(".?!,'\"\n ")
               if len(processed_sentence) > 0:
                 processed_sentence = processed_sentence+" "
               processed_sentence = processed_sentence+words[i]
          if processed_sentence != sentence:
            print "preprocessed: %s"%(processed_sentence)
            #print "original:     %s\npreprocessed: %s"%(sentence, processed_sentence)
          list_to_return.append(processed_sentence)
        list_of_lists_to_return.append(list_to_return)
    except Exception, e:
      print e
      print "Could not import NLTK. Please install it."
      return sentences_lists
    return list_of_lists_to_return
  else:
    return sentences_lists


def is_punctuation(char):
  return char in [".", "?", ",", "!", ")"]

def is_plural_pronoun(word):
  if len(word) > 0 and is_punctuation(word[-1]):
    word = word[:-1]
  return word.lower() in ["they", "their", "we", "our", "we're", "they're"]

def is_possessive_pronoun(word):
  if len(word) > 0 and is_punctuation(word[-1]):
    word = word[:-1]
  return word.lower() in ["his", "her", "its", "their"]
  

def is_pronoun_to_replace(word):
  if len(word) > 0 and is_punctuation(word[-1]):
    word = word[:-1]
  return word.lower() in ["he", "his", "she", "her", "s/he", "they", "their", "they're"]#, "it", "its"] 
  
