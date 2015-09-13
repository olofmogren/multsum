#!/usr/bin/python
# -*- coding: utf-8 -*-

import math, numpy, os.path, re, sys,time
from stemming.porter2 import stem
from multsum_clustering import *
from analyze_sentiment import analyze_sentiment
from w2v_client import *

 # @author Olof Mogren
 #
 # email: olof@mogren.one
 #
 # 
 # Base class for submodular summarization.
 # 
 # Providing some base functions.
 #
 # Released under the GNU GPL.
 # 
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 #
 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program.  If not, see <http://www.gnu.org/licenses/>.
 #

A                    = 5.0
DEFAULT_LAMBDA       = 6.0
LINBILMES_CONSTANT_R = 0.3
# ALPHA              = 0.75f #LIn-bilmes: alpha = a/N

UNIT_WORDS           = 1
UNIT_SENTENCES       = 2
UNIT_CHARACTERS      = 3

LENGTH_UNITS = dict()
LENGTH_UNITS['WORDS'] = UNIT_WORDS
LENGTH_UNITS['SENTENCES'] = UNIT_SENTENCES
LENGTH_UNITS['CHARACTERS'] = UNIT_CHARACTERS

DEFAULT_SUMMARY_LENGTH = 300

REGEX_SPACE         = "\\W+"
REGEX_SPACE_COMMA   = "\\s+|,"
REGEX_NONWORD       = "[^\\p{L}\\p{Nd}]+"

DEFAULT_STOPWORDS   = os.path.split(os.path.realpath(__file__))[0]+'/english_stopwords.txt'
W2V_VECTOR_FILE     = '/home/mogren/tmp/GoogleNews-vectors-negative300.bin'

WELCOME_MESSAGE     = '''Welcome to MULTSUM.

Some of the options below can also be used with the GUI for MULTSUM, multsum_gui.py.

1. To summarize a set of documents, run:

   'python multsum.py inputfile1 inputfile2 ...'

   Unless other things are specified with options, input files are assumed to be
   text files with one sentence per line. Documents will be summarized with a
   multiplicative aggregation of three similarity measures (See Mogren et.al. 2015):
   TFIDF, Positive Sentiment, Negative Sentiment, and Continuous Vector Space models.

   Options:

     --split:           Split sentences using regex.
     --no-tfidf:        Do not use similarity measure based on
                        tfidf (See Lin&Bilmes, 2011) (Default is to use it).
     --no-sentiment:    Do not use similarity measure based on
                        sentiment analysis (See Mogren et.al. 2015) (Default is to use it).
     --no-w2v:          Do not use similarity measure based on
                        continuous vector space models (See Mogren et.al. 2015) (Default is to use it).
                        (This option is available in multsum_gui.py).
     --w2v-file <path>: Specify where to find the bin-file with vectors from the word2vec tool.
                        Pretrained vectors can be downloaded from http://code.google.com/p/word2vec/ .
                        (This option is available in multsum_gui.py).
     --w2v-backend:     Try to connect to running backend providing word vectors. See w2v_worker.py.

2. To use the sentence selection with user specified similarity matrices, run:

   'python multsum.py --m matrixfile1 matrixfile2 ... [--s sentences_file]

   If sentences_file is not provided, will output line-numbers instead of the actual summary.
   Line-numbers in the output starts at number one!
   The dimension of the similarity matrices need to match the number of sentences in file.
   The file is assumed to be text files with one sentence per line.

Global Options:

   --summary-length <len>: Set summary length. (Default is 300 words).
   --sumary-length-unit WORDS|CHARACTERS|SENTENCES: Set summary length unit (Default is WORDS).

For questions, please contact olof@mogren.one.

'''

def L1(S, w, alpha, a):
  if not alpha:
    alpha = a/(1.0*w.shape[0])
  res = 0.0
  for i in range(0, w.shape[0]):
    sum_val = 0.0; sumV = 0.0
    for j in S:
      sum_val += w[i][j]
    for k in range(0,w.shape[0]):
      sumV += w[i][k]
    sumV *= alpha
    res += min(sum_val, sumV)
  return res

def R1(S, w, clustering, K):
  N = w.shape[0]
  res = 0.0
  for k in range(0, K):
    sum_val = 0.0
    for j in S:
      if (clustering [j] == k):
        # sumV is called r_j in paper.
        sumV = 0.0
        for i in range(0,N):
          sumV += w [i][j]
        sum_val += sumV / N
    res += math.sqrt(sum_val)
  return res

def normalize(m):
  max_v = 0.0
  mr = zeroes((m.shape[0],m.shape[1]))
  #Get the  max_v:
  for i in range(0, m.shape[0]):
    for j in range(0, m.shape[1]):
      if m[i][j] > max_v:
        max_v = m[i][j]

  #Normalize:
  for i in range(0, m.shape[0]):
    for j in range(0, m.shape[1]):
      mr[i][j] = m[i][j]/max_v
  return mr
  
def getMultipliedAggregateSimilarities(ms):
  if len(ms) == 1:
    return ms[0]

  aggregateSim = ms[0]

  for k in range(1,len(ms)):#skip first, it's already added above.
    m = ms[k]
    for i in range(0, m.shape[0]):
      for j in range(0, m.shape[1]):
        aggregateSim[i][j] *= m[i][j]

  return aggregateSim

def getSentencesAsBags(stopwords, sentencesLists):
  sentenceBagList = list()

  for sentences in sentencesLists:
    for sentence in sentences:
      currSentence = dict()
      if len(sentence) > 0:
        words = re.split(REGEX_SPACE, sentence)
        prev = None
        for w in words:
          if not w.replace("_", ""):
            continue

          w = w.lower()

          stemmed = stem(w)
          if prev:
            bigram = prev+" "+stemmed
            currSentence[bigram] = currSentence.get(bigram,0)+1
            #end, bigrams

          if w not in stopwords:
            currSentence[stemmed] = currSentence.get(stemmed,0)+1
            prev = stemmed
          else:
            prev = w

      sentenceBagList.append(currSentence)
  return sentenceBagList

def get_stopwords(stopwordsFilename):
  stopwords = list()
  f = open(stopwordsFilename, 'r')
  for line in f:
    stopwords.append(line)
  f.close()
  return stopwords

def count_sentences(sentencesLists):
  lines = 0
  for l in sentencesLists:
    lines += len(l)
  return lines

def get_sentence_index(i, sentencesLists):
  searchedLines = 0
  for l in sentencesLists:
    if searchedLines + len(l) > i:
      return l[i-searchedLines]
    else:
      searchedLines += len(l)
  return None

def charactersLength(selected, sentencesLists):
  chars = 0
  for i in selected:
    chars += len(get_sentence_index(i, sentencesLists))
  return chars

def wordsLength(selected, sentencesLists):
  words = 0
  for i in selected:
    words += len(get_sentence_index(i, sentencesLists).split())
    #words += len(re.split(REGEX_NONWORD, get_sentence_index(i, sentencesLists)))
  return words

def summary_is_too_short(selected, sentencesLists, lengthUnit, summarySize):
  if lengthUnit == UNIT_CHARACTERS:
    return charactersLength(selected, sentencesLists) < summarySize
  elif lengthUnit == UNIT_WORDS:
    return wordsLength(selected, sentencesLists) < summarySize
  else:
    return len(selected) < summarySize
def summary_is_too_long(selected, sentencesLists, lengthUnit, summarySize):
  if lengthUnit == UNIT_CHARACTERS:
    return charactersLength(selected, sentencesLists) > summarySize
  elif lengthUnit == UNIT_WORDS:
    return wordsLength(selected, sentencesLists) > summarySize
  else:
    return len(selected) > summarySize

def get_def_sentsims(sentencesLists, stopwordsFilename, idfs):
  stopwords = get_stopwords(stopwordsFilename) #set

  sentencesBags = getSentencesAsBags(stopwords, sentencesLists) #list of dicts, from word to wordcount
  #print(sentencesBags)

  vocabulary_s = set() #list of strings
  for sentence in sentencesBags:
    for term in sentence:
      vocabulary_s.add(term)

  vocabulary = list(vocabulary_s)
  vocabulary.sort()

  vocabularyIndices = dict()
  for i in range(0,len(vocabulary)):
    vocabularyIndices[vocabulary[i]] = i

  # creating arrays for containing sentence vectors
  # each row is a sentence, each column corresponds to a word.
  sentenceTFIDFVectors = numpy.zeros((len(sentencesBags),len(vocabulary)))
  sentenceIDFVectors = numpy.zeros((len(sentencesBags),len(vocabulary)))

  if not idfs:
      # The following is what lin-bilmes did, if sentencesLists contains each document in a cluster.
    idfs = getIdfsFromDocCollection(sentencesLists, stopwordsFilename)
  #print(idfs)
  
  # Denominators for the cosine similarity computation: #/
  tfidfden = numpy.zeros((len(sentencesBags)))
  idfden = numpy.zeros((len(sentencesBags)))
  for i in range(0, len(sentencesBags)):
    for term in sentencesBags[i]:
      tf = sentencesBags[i][term]
      idf = idfs.get(term,None)
      if not idf:
          #Ugly hack. Because of some mismatch in sentence splitting on DUC, you sometimes get idfs not found for some bigrams. Will treat as if present in one document.
        idf = 1.0
        idfs[term] = idf
        #print("No idf for "+term+"! ")
    
      if not tf:
        print("No tf for "+term+"! STRANGE!")
      #Double tfidf = ((1+Math.log10(tf))*idf) #manning coursera nlp-course
      tfidf = tf*idf #lin-bilmes paper.

      sentenceTFIDFVectors[i][vocabularyIndices[term]] = tfidf
      sentenceIDFVectors[i][vocabularyIndices[term]] = idf

      tfidfden[i] += tfidf * tfidf
      idfden[i] += idf * idf
    
    tfidfden[i] = math.sqrt(tfidfden[i])
    idfden[i] = math.sqrt(idfden[i])

  # Numerators for the cosine similarity computation: */
  tfidfsim = numpy.eye(len(sentencesBags))
  idfdist = numpy.zeros((len(sentencesBags),len(sentencesBags)))
  sentenceTFIDFEuclidean = numpy.zeros((len(sentencesBags),len(sentencesBags)))

  for i in range(0,len(sentencesBags)):
    for j in range(0,len(sentencesBags)):
      euclideanSum = 0.0; tfidfnum = 0.0; idfnum = 0.0
      for term in sentencesBags[i]:
        tf_i = sentencesBags[i].get(term,0)
        tf_j = sentencesBags[j].get(term,0)
        idf = idfs[term]
        if not idf:
          #Ugly hack. Because of some mismatch in sentence splitting on DUC, you sometimes get idfs not found for some bigrams. Will treat as if present in one document.
          idf = 1.0
          idfs[term] = idf
          print("No idf for "+term+"! ")

        euclideanSum += math.pow(tf_i*idf-tf_j*idf, 2)

        #tfidf =  ((1+Math.log10(tf))*idf) #manning coursera nlp-course
        tfidf_i = tf_i*idf #lin-bilmes paper.
        tfidf_j = tf_j*idf #lin-bilmes paper.
        tfidfnum += tfidf_i * tfidf_j
        idfnum += idf * idf
   
      if tfidfden[i]==0 or tfidfden[j]==0:
        tfidfsim[i][j] = tfidfsim[j][i] = 0.0
      else:
        tfidfsim[i][j] = tfidfsim[j][i] = tfidfnum / (tfidfden[i] * tfidfden[j])
      if idfden[i]==0 or idfden[j]==0:
        idfdist[i][j] = idfdist[j][i] = 1.0
      else:
        idfdist[i][j] = idfdist[j][i] = 1.0 - idfnum / (idfden[i] * idfden[j])
      sentenceTFIDFEuclidean[i][j] = sentenceTFIDFEuclidean[j][i] = math.sqrt(euclideanSum)
    

  ret_dict = dict()
  ret_dict["tfidf_cosine"] = tfidfsim
  ret_dict["tfidf_euclidean"] = sentenceTFIDFEuclidean
  ret_dict["idf_dist"] = idfdist
  ret_dict["idf_vectors"] = sentenceIDFVectors
  ret_dict["tfidf_vectors"] = sentenceTFIDFVectors

  #for i in range(0,sentenceIDFVectors.shape[0]):
  #  for j in range(0,sentenceIDFVectors.shape[1]):
  #    print(str(sentenceIDFVectors[i][j])+" ")
  #  print("\n")
  
  return ret_dict
  
def select_sentences(summarySize,
                     matrices,
                     sentenceVectors,
                     sentencesLists,
                     lengthUnit,
                     idfVectorFileName,
                     docName):
  #discarded = set()
  selected = set()
  aggMatrix = getMultipliedAggregateSimilarities(matrices)

  #for i in range(0,aggMatrix.shape[0]):
  #  for j in range(0,aggMatrix.shape[1]):
  #    print str(aggMatrix[i][j])+' ',
  #  print 'EOL'

  K = getK(count_sentences(sentencesLists))
  if not sentenceVectors is None:
    clustering = getClusteringByVectors(sentenceVectors, K, idfVectorFileName, docName)
  else:
    clustering = get_clustering(sentencesLists, DEFAULT_STOPWORDS)

  #print 'clustering:'
  #for i in clustering:
  #  print i

  while summary_is_too_short(selected, sentencesLists, lengthUnit, summarySize):
    max_val = 0.0
    argmax = None
    for i in range(0,aggMatrix.shape[0]):
      if i not in selected:# and i not in discarded:
        selected.add(i)
        curr = L1 (selected, aggMatrix, None, A) + DEFAULT_LAMBDA * R1(selected, aggMatrix, clustering, K)
        # as in Lin-Bilmes 2010: */
        #print(str(curr)+" "+str(max_val)+" "+str(argmax))
        #print(selected)
        if curr > max_val:
          argmax = i
          max_val = curr
        selected.remove(i)

    if argmax:
      selected.add(argmax) #internal: zero-based.
      #selectedList.add(argmax+1) #outside visibility: one-based indexing.
    else:
      break

  #print("zero-based set:  ")
  #print(selected)

  #MoD_SINGLETON:
  currentlyBestCScore = L1(selected, aggMatrix, None, A) + DEFAULT_LAMBDA * R1(selected, aggMatrix, clustering, K)
  currentlyBestSingleton = None
  for i in range(0,aggMatrix.shape[0]):
    singleton = set()
    singleton.add(i)
    if not summary_is_too_long(singleton, sentencesLists, lengthUnit, summarySize):
      singletonSummaryScore = L1(singleton, aggMatrix, None, A) + DEFAULT_LAMBDA * R1(singleton, aggMatrix, clustering, K)
      if singletonSummaryScore > currentlyBestCScore:
        currentlyBestCScore = singletonSummaryScore
        currentlyBestSingleton = i
     
  if currentlyBestSingleton:
    print("Using singleton!")
    selected = set()
    selected.add(currentlyBestSingleton)
  
  return selected
  
  #
   # As described by lin-bilmes.
   # 
   #/

def getIdfsFromDocCollection(documentCluster, stopwordsFilename):
  documentCountsForTerm = dict() # dictfrom string to integer
  stopwords = get_stopwords(stopwordsFilename)

  # Special case: if all sentences are from one file, idfs from one document make no sense.
  # (similarities would be zero).
  # Will then consider each sentence as one document.
  if len(documentCluster) <= 1:
    newCluster = list()
    document = documentCluster[0]
    for s in document:
      l = list()
      l.append(s)
      newCluster.append(l)
    
    documentCluster = newCluster
  

  for document in documentCluster:
    curDoc = set()
    for sentence in document:
      if len(sentence) > 0:
        words = sentence.split()
        prev = None
        for w in words:
          w = w.replace("_", "").replace(".", "").replace(",", "").replace("!", "").replace("?", "").replace("-", "")
          if not w:
            continue

          w = w.lower()
          stemmed = stem(w)

          # We always include bigrams when computing IDFS. These can be ignored later,
          # but Lin&Bilmes used them.
          if prev:
            curDoc.add(prev+" "+stemmed)

          if w not in stopwords:
            curDoc.add(stemmed)
            prev = stemmed
          else:
            prev = w

    for term in curDoc:
      documentCountsForTerm[term] = documentCountsForTerm.get(term, 0)+1
    
  
  idfs = dict()
  for term in documentCountsForTerm:
    idfs[term] = math.log(len(documentCluster)/documentCountsForTerm[term], 10)
  
  #print("Done calculating IDFS.")

  return idfs

def summarize_matrix_files(matrix_files, sentence_file=None, stopwords=DEFAULT_STOPWORDS, length=DEFAULT_SUMMARY_LENGTH, unit=UNIT_WORDS):
  matrices = list()

  for filename in matrix_files:
    values = []
    f = open(filename, 'r')
    for line in f:
      if line:
        row = []
        for val in line.strip().strip(';').split(','):
          #print val
          row.append(float(val))
        #print 'new line!'
        values.append(row)

  
    matrix = numpy.zeros((len(values),len(values)))
    for i in range(0, len(values)):
      for j in range(0, len(values)):
        matrix[i][j] = values[i][j]
    
    matrices.append(matrix)

  sentencesLists = list()
  print sentence_file
  f = open(sentence_file, 'r')
  sentences = list()
  for line in f:
    if line:
      sentences.append(line)
  sentencesLists.append(sentences)

  summary_set = select_sentences(length,
                     matrices,
                     None,
                     sentencesLists,
                     unit,
                     None,
                     'summarization_doc')
  summary_list = list(summary_set)
  summary_list.sort()
  return_string = ''

  print 'Summary:'
  for i in summary_list:
    if sentencesLists:
      return_string += get_sentence_index(i, sentencesLists)+'\n'
      print('  '+get_sentence_index(i, sentencesLists))
    else:
      print(i+1) #one-based output, not zero-based.
  return return_string

def get_sentence_rep(sentence, wordmodel, w2v_backend):
  words = re.split('\W+', sentence)
  sentence_rep = 0.0
  count = 0.0
  for w in words:
    if wordmodel:
      wordrep = None
      if w in wordmodel:
        wordrep = wordmodel[w]
      elif w2v_backend:
        wordrep = w2v_get_representation(w)

      if wordrep:
        sentence_rep += wordrep
        count = count + 1.0

  return numpy.divide(sentence_rep, count)


def summarize_strings(sentencesLists, stopwords=DEFAULT_STOPWORDS, length=DEFAULT_SUMMARY_LENGTH, unit=UNIT_WORDS, use_tfidf_similarity=True, use_sentiment_similarity=True, use_w2v_similarity=True, w2v_vector_file=W2V_VECTOR_FILE, split_sentences=False, preloaded_w2v_wordmodel=None, w2v_backend=False):

  print 'summarize_strings()'
  #for l in sentencesLists:
  #  for s in l:
  #    print s

  if split_sentences:
    print 'splitting'
    splittedLists = []
    for l in sentencesLists:
      splittedList = []
      for s in l:
        #splitted = re.split('[\.!?]', s)
        splitted = re.split('(?<=[\.!\?])\W+', s)
        for s in splitted:
          if s:
            splittedList.append(s.replace('\n', ''))
            #print s
      splittedLists.append(splittedList)
    sentencesLists = splittedLists
  
  sentsims = get_def_sentsims(sentencesLists, stopwords, None)

  matrices = list()
  flat_sentences = [sentence for document in sentencesLists for sentence in document]
  if use_sentiment_similarity:
    (pos, neg) = analyze_sentiment(flat_sentences)
    matrices.append(pos)
    matrices.append(neg)
  if use_w2v_similarity:
    print('Computing sentence similarities based on word2vec.')
    if preloaded_w2v_wordmodel:
      wordmodel = preloaded_w2v_wordmodel
    elif not w2v_client:
      wordmodel = load_w2v_wordmodel(w2v_vector_file)
    if wordmodel or w2v_backend:
      w2v_matrix = numpy.zeros((len(flat_sentences), len(flat_sentences)))
      for i in range(0, len(flat_sentences)):
        sentence_rep_i = get_sentence_rep(flat_sentences[i], wordmodel, w2v_backend)
        for j in range(i, len(flat_sentences)):
          sentence_rep_j = get_sentence_rep(flat_sentences[j], wordmodel, w2v_backend)
          w2v_matrix[i][j] = 0.5 * (numpy.dot(sentence_rep_i, sentence_rep_j)/numpy.sqrt(numpy.dot(sentence_rep_i, sentence_rep_i))+numpy.sqrt(numpy.dot(sentence_rep_j, sentence_rep_j))+1)
          w2v_matrix[j][i] = w2v_matrix[i][j]
      matrices.append(w2v_matrix)
  if use_tfidf_similarity or len(matrices) == 0:
    # this is also used for fallback if the others were specified and failed for some reason.
    matrices.append(sentsims["tfidf_cosine"])

  print 'Input sentences:'
  for l in sentencesLists:
    for s in l:
      print '  '+s
  print 'new list:'

  #for m in matrices:
  #  for i in range(0,m.shape[0]):
  #    for j in range(0,m.shape[1]):
  #      print str(m[i][j])+' ',
  #    print 'EOL'

  summary_set = select_sentences(length,
                     matrices,
                     sentsims["idf_vectors"],
                     sentencesLists,
                     unit,
                     None,
                     'summarization_doc')
  summary_list = list(summary_set)
  summary_list.sort()
  return_string = ''
  print 'Summary:'
  for i in summary_list:
    return_string += get_sentence_index(i, sentencesLists)+'\n'
    print('  '+get_sentence_index(i, sentencesLists))
  return return_string
 

def summarize_files(document_names, length=DEFAULT_SUMMARY_LENGTH, unit=UNIT_WORDS, use_tfidf_similarity=True, use_sentiment_similarity=True, use_w2v_similarity=True, split_sentences=False, w2v_vector_file=W2V_VECTOR_FILE, preloaded_w2v_wordmodel=None, w2v_backend=False):
  sentencesLists = list()
  for filename in document_names:
    f = open(filename, 'r')
    sentences = list()
    for line in f:
      if line:
        sentences.append(line)
    sentencesLists.append(sentences)
  
  return summarize_strings(sentencesLists, length=length, unit=unit, use_tfidf_similarity=use_tfidf_similarity, use_sentiment_similarity=use_sentiment_similarity, use_w2v_similarity=use_w2v_similarity, w2v_vector_file=w2v_vector_file, split_sentences=split_sentences, preloaded_w2v_wordmodel=preloaded_w2v_wordmodel, w2v_backend=w2v_backend)

def load_w2v_wordmodel(w2v_vector_file=W2V_VECTOR_FILE):
  if not os.path.isfile(w2v_vector_file):
    print('Word2Vec vector file not found! Looked in '+w2v_vector_file+'. Will go on without this similarity measure.')
    return None
  else:
    from gensim.models import word2vec
    statinfo = os.stat(w2v_vector_file)
    if statinfo.st_size > 1073741824:
      print('Loading word2vec file into memory. File is big (%d gigabytes). This might take a while. Run with --no-w2v to not use word2vec.'%(statinfo.st_size/1073741824.0))
    return word2vec.Word2Vec.load_word2vec_format(w2v_vector_file, binary=True)

def get_clustering(sentencesLists, stopwords=DEFAULT_STOPWORDS):
  sentsims = get_def_sentsims(sentencesLists, stopwords, None)
  K = getK(sentsims["tfidf_cosine"].shape[0])
  clustering = getClusteringByVectors(sentsims["idf_vectors"], K, None, "summarization_doc")
  return clustering


def main():
  print WELCOME_MESSAGE
  
  doc_files = True
  files = list()
  sentences_file = None
  skip = False

  split_sentences = False

  use_tfidf_similarity = True
  use_sentiment_similarity = True
  use_w2v_similarity = True

  summary_length = DEFAULT_SUMMARY_LENGTH
  summary_length_unit = UNIT_WORDS

  w2v_vector_file = W2V_VECTOR_FILE
  w2v_backend = False

  for i in range(1,len(sys.argv)):
    if skip:
      skip = False
      continue

    if sys.argv[i] == '--m':
      # matrix files
      doc_files = False
    elif sys.argv[i] == '--split':
      split_sentences = True
    elif sys.argv[i] == '--no-tfidf':
      use_tfidf_similarity = False
    elif sys.argv[i] == '--no-sentiment':
      use_sentiment_similarity = False
    elif sys.argv[i] == '--no-w2v':
      use_w2v_similarity = False
    elif sys.argv[i] == '--sumary-length':
      summary_length = sys.argv[i+1]
      skip = True
    elif sys.argv[i] == '--sumary-length-unit':
      summary_length_unit = LENGTH_UNITS[sys.argv[i+1]]
      skip = True
    elif sys.argv[i] == '--w2v-file':
      w2v_vector_file = sys.argv[i+1]
      skip = True
    elif sys.argv[i] == '--w2v-backend':
      w2v_backend = True
    elif sys.argv[i] == '--s':
      # matrix files
      sentences_file = sys.argv[i+1]
      skip = True
    else:
      files.append(sys.argv[i])
  
  if doc_files:
    if not use_tfidf_similarity and not use_sentiment_similarity and not use_w2v_similarity:
      print 'Using default LinTFIDF similarity measure, since no other was provided.'
      use_tfidf_similarity = True
    summarize_files(files, length=summary_length, unit=summary_length_unit, use_tfidf_similarity=use_tfidf_similarity, use_sentiment_similarity=use_sentiment_similarity, use_w2v_similarity=use_w2v_similarity, split_sentences=split_sentences, w2v_vector_file=w2v_vector_file, w2v_backend=w2v_backend)
  else:
    summarize_matrix_files(files, sentences_file)

if  __name__ =='__main__':main()

