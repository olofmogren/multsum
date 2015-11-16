import re

REGEX_SPACE         = " +"

def preprocess(documents, anaphora_resolution_simple=False, quiet=False):
  if anaphora_resolution_simple:
    documents_to_return = list()
    try:
      for document in documents:
        document_to_return = list()
        previous_person = None
        previous_person_plural = None
        for sentence in document:
          # sentence is already a list of words. No need to split.
          #words = re.split(REGEX_SPACE, sentence)
          new_entity = get_last_namedentity(sentence)
          if new_entity:
            if new_entity["type"] == "PERSON":
              previous_person = new_entity["entity"]
            else:
              previous_person_plural = new_entity["entity"]
          processed_sentence = list()
          for i in xrange(0, len(sentence)):
            #if (tags[i] == "PRP" or tags[i] == "PRP$") and words[i] != "it":
            if is_pronoun_to_replace(sentence[i]):
              possessive_suffix = ""
              if is_possessive_pronoun(sentence[i]):
              #if tags[i][-1] == "$":
                # English bias:
                possessive_suffix = "'s"
              punctuation = ""
              last_char = sentence[i][-1]
              word_without_punctuation = sentence[i]
              if is_punctuation(last_char):
                punctuation = last_char
                word_without_punctuation = sentence[i][:-1]
              if is_plural_pronoun(sentence[i]) and (previous_person_plural is not None):
                #These are usually crap.
                #processed_sentence.append(word_without_punctuation+" ("+previous_person_plural+possessive_suffix+")"+punctuation)
                processed_sentence.append(sentence[i])
              elif previous_person is not None:
                processed_sentence.append(word_without_punctuation+" ("+previous_person+possessive_suffix+")"+punctuation)
              else:
                processed_sentence.append(sentence[i])
            else:
            #   if tags[i] == "NNP":
            #     previous_person = sentence[i].strip(".?!,'\"\n ")
            #   elif tags[i] == "NNPS":
            #     previous_person_plural = sentence[i].strip(".?!,'\"\n ")
              processed_sentence.append(sentence[i])
          if processed_sentence != sentence and not quiet:
            print "preprocessed: %s"%(' '.join(processed_sentence))
            #print "original:     %s\npreprocessed: %s"%(sentence, processed_sentence)
          document_to_return.append(processed_sentence)
        documents_to_return.append(document_to_return)
    except Exception, e:
      print e
      print "Could not import NLTK. Please install it."
      return documents
    return documents_to_return
  else:
    return documents

def get_last_namedentity(sentence):
  '''
    go from internal (semeval-inspired) representation with dict
    fro each token, to a pair-representation used in python nltk. In this
    representation, words are index 0 and pos tags are index
  '''
  import nltk
  current_sentence_ner_chunked_tree = nltk.ne_chunk(nltk.pos_tag(sentence))

  # E.G. Tree('S', [('Hello', 'NNP'), ('I', 'PRP'), ('like', 'VBP'), ('printing', 'VBG'), ('stuff', 'NN'), ('with', 'IN'), Tree('ORGANIZATION', [('Capital', 'NNP')]), ('Letters.', 'NNP'), ('The', 'NNP'), Tree('ORGANIZATION', [('Capital', 'NNP')]), ('of', 'IN'), Tree('GPE', [('Sweden', 'NNP')]), ('is', 'VBZ'), Tree('PERSON', [('Stockholm', 'NNP')])])
  # Flatten (crates a list of dicts. Each dict corresponds to a token. It has keys: 'word' 'pos' and 'ne_index', the latter of which is -1 whenever the word does not belong to a NER).:

  current_sentence_ner_chunked = []

  for outer_index in range(0, len(current_sentence_ner_chunked_tree)):
    if getattr(current_sentence_ner_chunked_tree[outer_index], 'node', None) is not None:
      #in ['ORGANIZATION', 'GPE', 'PERSON', 'LOCALIZATION']:
      ne_len = len(current_sentence_ner_chunked_tree[outer_index])
      for ne_index in range(0, ne_len):
        token = {'word': current_sentence_ner_chunked_tree[outer_index][ne_index][0], 'pos': current_sentence_ner_chunked_tree[outer_index][ne_index][1], 'ne_index': ne_index, 'ne_len': ne_len, 'ne_type': current_sentence_ner_chunked_tree[outer_index].node}
        current_sentence_ner_chunked.append(token)
    else:
      token = {'word': current_sentence_ner_chunked_tree[outer_index][0], 'pos': current_sentence_ner_chunked_tree[outer_index][1], 'ne_index': -1, 'ne_len': 0, 'ne_type': None}
      current_sentence_ner_chunked.append(token)
  #print str(current_sentence_ner_chunked)
  if len(sentence) != len(current_sentence_ner_chunked):
    raise Exception('len(sentence): %d, len(current_sentence_ner_chunked): %d'%(len(sentence), len(current_sentence_ner_chunked)))

  entity = None
  for i in range(0, len(current_sentence_ner_chunked)):
    token = current_sentence_ner_chunked[i]
    if token["ne_index"] == 0 and token["ne_type"] in ["PERSON", "GPE", "ORGANIZATION"]:
      entity = {"entity": ' '.join([t["word"] for t in current_sentence_ner_chunked[i:i+token["ne_len"]]]), "type": token["ne_type"]}
  #if entity:
  #  print "found entity: %s"%entity
  return entity

def preprocess_old(documents, anaphora_resolution_simple=False, quiet=False):
  if anaphora_resolution_simple:
    documents_to_return = list()
    try:
      for document in documents:
        document_to_return = list()
        previous_person = None
        previous_person_plural = None
        for sentence in document:
          # sentence is already a list of words. No need to split.
          #words = re.split(REGEX_SPACE, sentence)
          import nltk
          tagged = nltk.pos_tag(sentence)
          tags = [item[1] for item in tagged]
          if len(tagged) != len(sentence):
            print "length of tagged (%d) differs from number of words in sentence (%d)"%(len(tagged) != len(words_i))
            exit()
          processed_sentence = list()
          for i in xrange(0, len(sentence)):
            #if (tags[i] == "PRP" or tags[i] == "PRP$") and words[i] != "it":
            if is_pronoun_to_replace(sentence[i]):
              possessive_suffix = ""
              if is_possessive_pronoun(sentence[i]):
              #if tags[i][-1] == "$":
                # English bias:
                possessive_suffix = "'s"
              punctuation = ""
              last_char = sentence[i][-1]
              word_without_punctuation = sentence[i]
              if is_punctuation(last_char):
                punctuation = last_char
                word_without_punctuation = sentence[i][:-1]
              if is_plural_pronoun(sentence[i]) and previous_person_plural is not None:
                processed_sentence.append(word_without_punctuation+" ("+previous_person_plural+possessive_suffix+")"+punctuation)
              elif previous_person is not None:
                processed_sentence.append(word_without_punctuation+" ("+previous_person+possessive_suffix+")"+punctuation)
            else:
               if tags[i] == "NNP":
                 previous_person = sentence[i].strip(".?!,'\"\n ")
               elif tags[i] == "NNPS":
                 previous_person_plural = sentence[i].strip(".?!,'\"\n ")
               processed_sentence.append(sentence[i])
          if processed_sentence != sentence and not quiet:
            print "preprocessed: %s"%(' '.join(processed_sentence))
            #print "original:     %s\npreprocessed: %s"%(sentence, processed_sentence)
          document_to_return.append(processed_sentence)
        documents_to_return.append(document_to_return)
    except Exception, e:
      print e
      print "Could not import NLTK. Please install it."
      return documents
    return documents_to_return
  else:
    return documents


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
  
