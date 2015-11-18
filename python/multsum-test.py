#!/usr/bin/python
# -*- coding: utf-8 -*-

import math, numpy, os.path, re, sys,time
import multsum_preprocess
from stemming.porter2 import stem
from multsum_clustering import *
from analyze_sentiment import analyze_sentiment
from backend_client import *

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

MIN_SENTENCE_LENGTH = 4

REGEX_SPACE         = "\\W+"
REGEX_SPACE_COMMA   = "\\s+|,"
REGEX_NONWORD       = "[^\\p{L}\\p{Nd}]+"

DEFAULT_STOPWORDS   = os.path.split(os.path.realpath(__file__))[0]+'/english_stopwords.txt'
W2V_VECTOR_FILE     = '/home/mogren/tmp/GoogleNews-vectors-negative300.bin'

WELCOME_MESSAGE     = '''Welcome to MULTSUM.

Some of the options below can also be used with the GUI for MULTSUM, multsum_gui.py.

1. To summarize a set of documents, run:

   'python multsum.py [options] inputfile1 inputfile2 ...'

   Unless other things are specified with options, input files are assumed to be
   text files with one sentence per line. Documents will be summarized with a
   multiplicative aggregation of three similarity measures (See Mogren et.al. 2015):
   TFIDF, Positive Sentiment, Negative Sentiment, and Continuous Vector Space models.

   Options:

     --spl:             Do not split sentences using regex. Assume one sentence per line instead (SPL).
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
     --w2v-experiments <experiment-tags>:
                        Run experimental versions of the w2v sentence similarity measure.
                        experiment-tags is a comma-separated list (without space) of:
                        TOP5 - using the average of the top 5 scoring word-similarities between the two sentences.
                       
                        

2. To use the sentence selection with user specified similarity matrices, run:

   'python multsum.py --m matrixfile1 matrixfile2 ... [--s sentences_file]

   If sentences_file is not provided, will output line-numbers instead of the actual summary.
   Line-numbers in the output starts at number one!
   The dimension of the similarity matrices need to match the number of sentences in file.
   The file is assumed to be text files with one sentence per line.

Global Options:

   --summary-length <len>:         Set summary length. (Default is 300 words).
   --summary-length-unit WORDS|CHARACTERS|SENTENCES:
                                   Set summary length unit (Default is WORDS).

   --use-aggregate-for-clustering; Cluster by aggregated similarities. Default is using tfidf ala Lin&Bilmes.

   --quiet:                        Do not talk so much.
   --numerical:                    Output line numbers instead of readable text.

   --input-limit <num-sentences>:  Input limit. If input is longer than this, will take a random sample as input.

For questions, please contact olof@mogren.one. I will answer after capacity.

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

  aggregateSim = numpy.array(ms[0], copy=True)

  for k in range(1,len(ms)):#skip first, it's already added above.
    m = ms[k]
    for i in range(0, m.shape[0]):
      for j in range(0, m.shape[1]):
        aggregateSim[i][j] *= m[i][j]

  minval = 1.0
  maxval = 0.0
  for i in range(0, aggregateSim.shape[0]):
    for j in range(0, aggregateSim.shape[1]):
      if aggregateSim[i][j] < minval:
        minval = aggregateSim[i][j]
      if aggregateSim[i][j] > maxval:
        maxval = aggregateSim[i][j]

  aggregateSim = (aggregateSim-minval)/(maxval-minval)

  return aggregateSim

def get_sentences_bags_flat(stopwords, documents):
  sentence_bag_list = list()

  first = True
  for document in documents:
    for sentence in document:
      #if first:
      #  print 'First sentence, (creating bags).'
      #  print sentence
      current_sentence = dict()
      if len(sentence) > 0:
        #words = re.split(REGEX_SPACE, sentence)
        # Input is already split. It is now a list of words.
        prev = None
        for w in sentence:
          w = filter(str.isalnum, w)
          if not w:
            continue
          stemmed = stem(w)
          if prev:
            bigram = prev+" "+stemmed
            current_sentence[bigram] = current_sentence.get(bigram,0)+1
            #end, bigrams

          if w not in stopwords:
            current_sentence[stemmed] = current_sentence.get(stemmed,0)+1
            prev = stemmed
          else:
            prev = w

      sentence_bag_list.append(current_sentence)
      first = False
  return sentence_bag_list

def get_stopwords(stopwordsFilename):
  stopwords = list()
  f = open(stopwordsFilename, 'r')
  for line in f:
    stopwords.append(line)
  f.close()
  return stopwords

def count_sentences(documents):
  lines = 0
  for document in documents:
    lines += len(document)
  return lines

def get_sentence_index(i, documents):
  searchedLines = 0
  for document in documents:
    if searchedLines + len(document) > i:
      return document[i-searchedLines]
    else:
      searchedLines += len(document)
  return None

def characters_length(selected, documents):
  chars = 0
  for i in selected:
    sentence_chars = 0
    sentence = get_sentence_index(i, documents)
    for word in sentence:
      # The +1 counts an implicit space between each word.
      sentence_chars += len(word)+1
    chars += sentence_chars
  return chars

def words_length(selected, documents):
  words = 0
  for i in selected:
    words += len(get_sentence_index(i, documents))
  return words

def summary_is_too_short(selected, documents, lengthUnit, summarySize):
  if lengthUnit == UNIT_CHARACTERS:
    return characters_length(selected, documents) < summarySize
  elif lengthUnit == UNIT_WORDS:
    return words_length(selected, documents) < summarySize
  else:
    return len(selected) < summarySize

def summary_is_too_long(selected, documents, lengthUnit, summarySize):
  if lengthUnit == UNIT_CHARACTERS:
    return characters_length(selected, documents) > summarySize
  elif lengthUnit == UNIT_WORDS:
    return words_length(selected, documents) > summarySize
  else:
    return len(selected) > summarySize

def get_def_sentsims(documents, stopwordsFilename, idfs):
  stopwords = get_stopwords(stopwordsFilename) #set

  sentences_bags = get_sentences_bags_flat(stopwords, documents) #list of dicts, from word to wordcount
  #print(sentences_bags)

  vocabulary_s = set() #list of strings
  for sentence in sentences_bags:
    for term in sentence:
      vocabulary_s.add(term)

  vocabulary = list(vocabulary_s)
  vocabulary.sort()

  #print vocabulary

  vocabularyIndices = dict()
  for i in range(0,len(vocabulary)):
    vocabularyIndices[vocabulary[i]] = i

  # creating arrays for containing sentence vectors
  # each row is a sentence, each column corresponds to a word.
  sentenceTFIDFVectors = numpy.zeros((len(vocabulary),len(sentences_bags)))
  sentenceIDFVectors = numpy.zeros((len(vocabulary),len(sentences_bags)))

  if not idfs:
      # The following is what lin-bilmes did, if documents contains each document in a cluster.
    idfs = get_idfs_from_doc_collection(documents, stopwords)
  #print(idfs)
  
  # Denominators for the cosine similarity computation: #/
  tfidfden = numpy.zeros((len(sentences_bags)))
  idfden = numpy.zeros((len(sentences_bags)))
  for i in range(0, len(sentences_bags)):
    for term in sentences_bags[i]:
      tf = sentences_bags[i][term]
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

      sentenceTFIDFVectors[vocabularyIndices[term]][i] = tfidf
      sentenceIDFVectors[vocabularyIndices[term]][i] = idf

      tfidfden[i] += tfidf * tfidf
      idfden[i] += idf * idf
    
    tfidfden[i] = math.sqrt(tfidfden[i])
    idfden[i] = math.sqrt(idfden[i])

  # Numerators for the cosine similarity computation: */
  tfidfsim = numpy.eye(len(sentences_bags))
  idfdist = numpy.zeros((len(sentences_bags),len(sentences_bags)))
  sentenceTFIDFEuclidean = numpy.zeros((len(sentences_bags),len(sentences_bags)))

  for i in range(0,len(sentences_bags)):
    for j in range(0,len(sentences_bags)):
      euclideanSum = 0.0; tfidfnum = 0.0; idfnum = 0.0
      for term in sentences_bags[i]:
        tf_i = sentences_bags[i].get(term,0)
        tf_j = sentences_bags[j].get(term,0)
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
                     documents,
                     lengthUnit,
                     idfVectorFileName,
                     docName,
                     use_aggregate_for_clustering = False):
  #discarded = set()
  selected = set()
  aggMatrix = getMultipliedAggregateSimilarities(matrices)

  #for i in range(0,aggMatrix.shape[0]):
  #  for j in range(0,aggMatrix.shape[1]):
  #    print str(aggMatrix[i][j])+' ',
  #  print 'EOL'

  #print "Computing clustering..."
  K = getK(count_sentences(documents))
  if use_aggregate_for_clustering:
    clustering = get_clustering(documents, DEFAULT_STOPWORDS, aggMatrix)
  elif not sentenceVectors is None:
    clustering = get_clustering_by_vectors(sentenceVectors, K, idfVectorFileName, docName)
  else:
    clustering = get_clustering(documents, DEFAULT_STOPWORDS)
  #print "Clustering done."

  #print 'clustering done. back in multsum now.'
  #for i in clustering:
  #  print i

  #print "Optimization loop:"
  while summary_is_too_short(selected, documents, lengthUnit, summarySize):
    max_val = 0.0
    argmax = None
    for i in range(0,aggMatrix.shape[0]):
      if i not in selected:# and i not in discarded:
        selected.add(i)
        curr = L1 (selected, aggMatrix, None, A) + DEFAULT_LAMBDA * R1(selected, aggMatrix, clustering, K)
        # as in Lin-Bilmes 2010: */
        #print(str(curr)+" "+str(max_val)+" "+str(argmax))
        if curr > max_val:
          argmax = i
          max_val = curr
        selected.remove(i)

    if argmax:
      selected.add(argmax) #internal: zero-based.
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
    if not summary_is_too_long(singleton, documents, lengthUnit, summarySize):
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

def get_idfs_from_doc_collection(documents, stopwords):
  documentCountsForTerm = dict() # dictfrom string to integer

  # Special case: if all sentences are from one file, idfs from one document make no sense.
  # (similarities would be zero).
  # Will then consider each sentence as one document.
  if len(documents) <= 1:
    newCluster = list()
    document = documents[0]
    for s in document:
      l = list()
      l.append(s)
      newCluster.append(l)
    
    documents = newCluster
  

  for document in documents:
    curDoc = set()
    for sentence in document:
      if len(sentence) > 0:
        #No need to split sentence. It's already a list of words/tokens
        #words = sentence.split()
        prev = None
        for w in sentence:
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
    idfs[term] = math.log(len(documents)/documentCountsForTerm[term], 10)
  
  #print("Done calculating IDFS.")

  return idfs

def summarize_matrix_files(matrix_files, sentence_file=None, stopwordsFilename=DEFAULT_STOPWORDS, length=DEFAULT_SUMMARY_LENGTH, unit=UNIT_WORDS, output_numbers=True, use_aggregate_for_clustering=False, quiet=False):
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

  documents = list()
  print sentence_file
  f = open(sentence_file, 'r')
  sentences = list()
  for line in f:
    if line:
      #words = re.split(REGEX_SPACE, line)
      words = line.split()
      sentences.append(words)
  documents.append(sentences)

  summary_set = select_sentences(length,
                     matrices,
                     None,
                     documents,
                     unit,
                     None,
                     'summarization_doc',
                     use_aggregate_for_clustering=use_aggregate_for_clustering)
  summary_list = list(summary_set)
  summary_list.sort()
  return_string = ''

  if not quiet:
    print 'Summary:'
  for i in summary_list:
    if output_numbers or not documents:
      return_string += (i+1)+'\n'
      if not quiet:
        print(i+1) #one-based output, not zero-based.
    else:
      return_string += get_sentence_index(i, documents)+'\n'
      if not quiet:
        print('  '+get_sentence_index(i, documents))
  return return_string

def get_sentence_embedding_avg(sentence, wordmodel, w2v_backend, quiet=False):
  # Sentence is a list of words/tokens. No need to split.
  #words = re.split(REGEX_SPACE, sentence.lower())
  sentence_embedding = 0.0
  count = 0.0
  for w in sentence:
    word = w.lower()
    word = filter(str.isalnum, word)
    wordrep = get_word_embedding(word, wordmodel, w2v_backend, quiet=quiet)
    if wordrep is None:
      wordrep = get_word_embedding(stem(word), wordmodel, w2v_backend, quiet=quiet)
    if wordrep is not None:
      sentence_embedding += wordrep
      count = count + 1.0

  if count == 0:
    return None
  return numpy.divide(sentence_embedding, count)

def get_word_embedding(word, wordmodel, w2v_backend, quiet=False):
  wordrep = None
  if wordmodel and word in wordmodel:
    wordrep = wordmodel[word]
  elif w2v_backend:
    wordrep = backend_get_representation(word, quiet=quiet)
  return wordrep

def get_w2v_matrix(flat_sentences, wordmodel, w2v_backend, stopwords, documents, w2v_experiments="", quiet=False):
  w2v_matrix = numpy.zeros((len(flat_sentences), len(flat_sentences)))
  minval = 1.0
  maxval = 0.0
  argmin_i = 0
  argmin_j = 0
  argmax_i = 0
  argmax_j = 0
  if w2v_experiments=="":
    for i in range(0, len(flat_sentences)):
      sentence_embedding_i = get_sentence_embedding_avg(flat_sentences[i], wordmodel, w2v_backend, quiet=quiet)
      for j in range(i, len(flat_sentences)):
        sentence_embedding_j = get_sentence_embedding_avg(flat_sentences[j], wordmodel, w2v_backend, quiet=quiet)
        score = 0.0
        if not (sentence_embedding_i is None or sentence_embedding_j is None):
          score = 0.5 * (numpy.dot(sentence_embedding_i, sentence_embedding_j)/numpy.sqrt(numpy.dot(sentence_embedding_i, sentence_embedding_i))*numpy.sqrt(numpy.dot(sentence_embedding_j, sentence_embedding_j))+1)
        w2v_matrix[i][j] = score
        w2v_matrix[j][i] = w2v_matrix[i][j]
        if w2v_matrix[i][j] < minval:
          minval = w2v_matrix[i][j]
          argmin_i = i
          argmin_j = j
        if w2v_matrix[i][j] > maxval:
          maxval = w2v_matrix[i][j]
          argmax_i = i
          argmax_j = j
  else:
    sentences_bags = get_sentences_bags_flat(stopwords, documents) #list of dicts, from word to wordcount
    idfs = get_idfs_from_doc_collection(documents, stopwords)
    #splitted_sentence_lists = list()
    selected_reps = list()
    reps = list()
    tfidfs = list()
    postagged_sentences = list()
    num_sentences = len(flat_sentences)
    for i in range(0, num_sentences):
      #if not quiet:
      # print "%d of %d (suppress with --quiet)."%(i, num_sentences)
      #words_i = re.split(REGEX_SPACE, flat_sentences[i].lower())
      #splitted_sentence_lists.append(words_i)
      if "POSMATCH" in w2v_experiments:
        try:
          import nltk
          tagged = nltk.pos_tag(flat_sentences[i])
          if len(tagged) != len(flat_sentences[i]):
            print "length of tagged (%d) differs from number of words in sentence (%d)"%(len(tagged) != len(flat_sentences[i]))
            exit()
          postagged_sentences.append([item[1] for item in tagged])
        except:
          print "Could not import NLTK. Please install it."
          exit()
      reps_i = list()
      tfidfs_i = list()
      for word_i in flat_sentences[i]:
        word_lc = filter(str.isalnum, word_i).lower()
        stem_i = stem(word_lc)
        if stem_i and stem_i in sentences_bags[i] and stem_i in idfs:
          tfidf_i = sentences_bags[i][stem_i]*idfs[stem_i]
        elif word_lc and word_lc in sentences_bags[i] and word_lc in idfs:
          tfidf_i = sentences_bags[i][word_lc]*idfs[word_lc]
        else:
          tfidf_i = 0.0 #TODO: will this work?
        word_embedding_i = get_word_embedding(word_lc, wordmodel, w2v_backend, quiet=quiet)
        if word_embedding_i is None:
          word_embedding_i = get_word_embedding(stem(word_lc), wordmodel, w2v_backend, quiet=quiet)
        #if word_embedding_i is None:
        #  continue
        if "TFIDFWEIGHT" in w2v_experiments:
          word_embedding_i *= tfidf_i
        tfidfs_i.append(tfidf_i)
        reps_i.append(word_embedding_i)
      tfidfs.append(tfidfs_i)
      reps.append(reps_i)
      if "ALLSELECT" in w2v_experiments or "POSMATCH" in w2v_experiments:
        selected_reps.append(reps_i)
      elif "TFIDFSELECT" in w2v_experiments:
        # SELECT THE ONES WITH HIGHEST TFIDF:
        indices = sorted(range(len(tfidfs_i)), key=tfidfs_i.__getitem__, reverse=True)[0:5]
        selected_reps.append([reps_i[index] for index in indices])
      else:
        print "Unknown w2v_experiment: \""+w2v_experiments+"\"."
        return None
    for i in range(0, len(flat_sentences)):
      for j in range(i+1, len(flat_sentences)):
        scores = []
        for ii in range(0,len(selected_reps[i])):
          for jj in range(0,len(selected_reps[j])):
            word_embedding_i = selected_reps[i][ii]
            word_embedding_j = selected_reps[j][jj]
            if "POSMATCH" in w2v_experiments and postagged_sentences[i][ii] != postagged_sentences[j][jj]:
              continue
            if word_embedding_i is None or word_embedding_j is None:
              continue
            similarity = 0.5 * (numpy.dot(word_embedding_i, word_embedding_j)/numpy.sqrt(numpy.dot(word_embedding_i, word_embedding_i))*numpy.sqrt(numpy.dot(word_embedding_j, word_embedding_j))+1)
            scores.append(similarity)
        sentence_sim = 0.0
        for k in range(0,len(scores)):
          #print scores[k]
          sentence_sim += scores[k]
          #print sentence_sim
        if len(scores) > 0:
          #divide-by-zero check. If zero, also sentence_sim will be zero
          sentence_sim /= float(len(scores))
        #print "%d,%d: %f"%(i,j,sentence_sim)
        w2v_matrix[i][j] = sentence_sim
        w2v_matrix[j][i] = w2v_matrix[i][j]
        if sentence_sim < minval:
          minval = sentence_sim
          argmin_i = i
          argmin_j = j
        if sentence_sim > maxval:
          maxval = sentence_sim
          argmax_i = i
          argmax_j = j


  for i in range(0, len(flat_sentences)):
    for j in range(i+1, len(flat_sentences)):
      normalized = (w2v_matrix[i][j]-minval)/(maxval-minval)
      w2v_matrix[i][j] = normalized
      w2v_matrix[j][i] = normalized
      #print "%d,%d: %f"%(i,j,normalized)
  # zero on diagonaL:
  for i in range(0, len(flat_sentences)):
    w2v_matrix[i][i] = 0.0

  
  if not quiet:
    print "maximally similar sentences (%d,%d) (score %f): \n  \"%s\"\n  \"%s\""%(argmax_i,argmax_j,w2v_matrix[argmax_i][argmax_j],' '.join(flat_sentences[argmax_i]), ' '.join(flat_sentences[argmax_j]))
    print "minimally similar sentences (%d,%d) (score %f): \n  \"%s\"\n  \"%s\""%(argmin_i,argmin_j,w2v_matrix[argmin_i][argmin_j],' '.join(flat_sentences[argmin_i]), ' '.join(flat_sentences[argmin_j]))
    #for i in range(0, len(flat_sentences)):
    #  for j in range(0, len(flat_sentences)):
    #    if j != 0:
    #      print ", ",
    #    print w2v_matrix[i][j],
    #  print ";"
  
  return w2v_matrix

'''
  input (documents) here is a list of lists of lists.
  Each document is a list of sentences.
  Each sentence is a list of words or tokens.

'''
def summarize_documents(documents, stopwordsFilename=DEFAULT_STOPWORDS, length=DEFAULT_SUMMARY_LENGTH, unit=UNIT_WORDS, use_tfidf_similarity=True, use_sentiment_similarity=True, use_w2v_similarity=True, w2v_vector_file=W2V_VECTOR_FILE, preloaded_w2v_wordmodel=None, w2v_backend=False, w2v_experiments="", quiet=False, output_numbers=False, use_aggregate_for_clustering=False, anaphora_resolution_simple=False, min_sentence_length=MIN_SENTENCE_LENGTH):

  # print 'summarize_documents()'
  #for l in documents:
  #  for s in l:
  #    print s
  documents = multsum_preprocess.preprocess(documents, anaphora_resolution_simple=anaphora_resolution_simple, quiet=quiet)
  
  sentsims = get_def_sentsims(documents, stopwordsFilename, None)

  matrices = list()
  flat_sentences = [sentence for document in documents for sentence in document]
  #for sentence in flat_sentences:
  #  print sentence
  if use_sentiment_similarity:
    (pos, neg) = analyze_sentiment(flat_sentences)
    matrices.append(pos)
    matrices.append(neg)
  if use_w2v_similarity:
    if not quiet:
      print('Computing sentence similarities based on word2vec.')
    wordmodel = None
    if preloaded_w2v_wordmodel:
      wordmodel = preloaded_w2v_wordmodel
    elif not w2v_backend:
      wordmodel = load_w2v_wordmodel(w2v_vector_file)
    if wordmodel or w2v_backend:
      w2v_matrix = get_w2v_matrix(flat_sentences, wordmodel, w2v_backend, get_stopwords(stopwordsFilename), documents, w2v_experiments, quiet=quiet)
      if not w2v_matrix is None:
        matrices.append(w2v_matrix)
  if use_tfidf_similarity or len(matrices) == 0:
    # this is also used for fallback if the others were specified and failed for some reason.
    matrices.append(sentsims["tfidf_cosine"])

  if not quiet:
    print 'Input sentences:'
    for l in documents:
      for s in l:
        print '  '+' '.join(s)

  #for m in matrices:
  #  for i in range(0,m.shape[0]):
  #    for j in range(0,m.shape[1]):
  #      print str(m[i][j])+' ',
  #    print 'EOL'

  summary_set = select_sentences(length,
                     matrices,
                     sentsims["idf_vectors"],
                     documents,
                     unit,
                     None,
                     'summarization_doc',
                     use_aggregate_for_clustering=use_aggregate_for_clustering)
  summary_list = list(summary_set)
  summary_list.sort()
  return_string = ''
  #if not quiet:
  #  print 'Summary:'
  for i in summary_list:
    if output_numbers:
      #print "outputting numbers: %d"%(i+1)
      return_string += "%d\n"%(i+1)
    else:
      return_string += ' '.join(get_sentence_index(i, documents))+'\n'
    #if not quiet:
    #  print('  '+get_sentence_index(i, documents))
  return return_string
 

def summarize_files(document_names, length=DEFAULT_SUMMARY_LENGTH, unit=UNIT_WORDS, use_tfidf_similarity=True, use_sentiment_similarity=True, use_w2v_similarity=True, split_sentences=True, w2v_vector_file=W2V_VECTOR_FILE, preloaded_w2v_wordmodel=None, w2v_backend=False, w2v_experiments="", quiet=False, output_numbers=False, use_aggregate_for_clustering=False, anaphora_resolution_simple=False, min_sentence_length=MIN_SENTENCE_LENGTH, input_limit=None):
  lines_lists = list()
  num_lines = 0
  for filename in document_names:
    f = open(filename, 'r')
    lines = list()
    for line in f:
      stripped = line.strip()
      if stripped:
        lines.append(stripped)
    num_lines += len(lines)
    lines_lists.append(lines)
    f.close()

  if not quiet:
    print "Total line count (suppress this with --quiet): %d."%num_lines
  

  sentence_count = 0
  documents = []
  if split_sentences:
    if not quiet:
      print 'splitting'
    for l in lines_lists:
      document = []
      documentstring = ""
      for s in l:
        documentstring += " \n"+s
      #splitted = re.split('[\.!?]', s)
      splitted = re.split('(?<=[\.!\?])\W+', documentstring)
      for s in splitted:
        stripped = s.strip()
        if stripped:
          # The following line splits the sentence into a list of words
          stripped = stripped.replace('\n', ' ')
          #wordlist = re.split(REGEX_SPACE, stripped)
          wordlist = stripped.split()
          #wordlist = stripped.split()
          if len(wordlist) >= min_sentence_length:
            sentence_count += 1
            document.append(wordlist)
      documents.append(document)
  else:
    for l in lines_lists:
      document = list()
      for s in l:
        stripped = s.strip()
        # The following line splits the sentence into a list of words
        #wordlist = re.split(REGEX_SPACE, stripped)
        wordlist = stripped.split()
        if len(wordlist) >= min_sentence_length:
          sentence_count += 1
          document.append(wordlist)
      documents.append(document)

  if not quiet:
    print "Total sentence count after min length filtering and (possibly) splitting: %d"%(sentence_count)
  if input_limit and sentence_count > input_limit:
    print "sampling %d sentences to summarize from. (Affecting flag: --input-limit)"%input_limit
    l = [sentence for document in documents for sentence in document]
    sample = random.sample(l, 500)
    documents = [sample]

  return summarize_documents(documents, length=length, unit=unit, use_tfidf_similarity=use_tfidf_similarity, use_sentiment_similarity=use_sentiment_similarity, use_w2v_similarity=use_w2v_similarity, w2v_vector_file=w2v_vector_file, preloaded_w2v_wordmodel=preloaded_w2v_wordmodel, w2v_backend=w2v_backend, w2v_experiments=w2v_experiments, quiet=quiet, output_numbers=output_numbers, use_aggregate_for_clustering=use_aggregate_for_clustering, anaphora_resolution_simple=anaphora_resolution_simple, min_sentence_length=min_sentence_length)

def load_w2v_wordmodel(w2v_vector_file=W2V_VECTOR_FILE):
  if not os.path.isfile(w2v_vector_file):
    print('Word2Vec vector file not found! Looked in '+w2v_vector_file+'. Will go on without this similarity measure.')
    return None
  else:
    from gensim.models import word2vec
    statinfo = os.stat(w2v_vector_file)
    if statinfo.st_size > 1073741824:
      print('Loading word2vec file into memory. File is big (%d gigabytes). This might take a while. Run with --no-w2v to not use word2vec, or use backend_worker.py.'%(statinfo.st_size/1073741824.0))
    return word2vec.Word2Vec.load_word2vec_format(w2v_vector_file, binary=True)

def get_clustering(documents, stopwordsFilename=DEFAULT_STOPWORDS, sentsim_matrix=None):
  if sentsim_matrix is None:
    sentsims = get_def_sentsims(documents, stopwordsFilename, None)
    sentsim_matrix = sentsims["tfidf_cosine"]
  # for outputting clusters:
  flat_sentences = [sentence for document in documents for sentence in document]
  K = getK(sentsim_matrix.shape[0])
  clustering = get_clustering_by_similarities(sentsim_matrix, K, "summarization_doc", debug_sentences=flat_sentences)
  return clustering


def main():
  doc_files = True
  files = list()
  sentences_file = None
  skip = False

  split_sentences = True

  use_tfidf_similarity = True
  use_sentiment_similarity = True
  use_w2v_similarity = True

  use_aggregate_for_clustering = False

  summary_length = DEFAULT_SUMMARY_LENGTH
  summary_length_unit = UNIT_WORDS

  w2v_vector_file = W2V_VECTOR_FILE
  w2v_backend = False
  w2v_experiments = ""

  quiet = False
  output_numbers = False

  anaphora_resolution_simple = False

  input_limit = None

  min_sentence_length = MIN_SENTENCE_LENGTH
  
  for i in range(1,len(sys.argv)):
    if skip:
      skip = False
      continue

    if sys.argv[i] == '--m':
      # matrix files
      doc_files = False
    elif sys.argv[i] == '--spl':
      split_sentences = False
    elif sys.argv[i] == '--anaphora-resolution-simple':
      anaphora_resolution_simple = True
    elif sys.argv[i] == '--no-tfidf':
      use_tfidf_similarity = False
    elif sys.argv[i] == '--no-sentiment':
      use_sentiment_similarity = False
    elif sys.argv[i] == '--no-w2v':
      use_w2v_similarity = False
    elif sys.argv[i] == '--input-limit':
      input_limit = int(sys.argv[i+1])
      skip = True
    elif sys.argv[i] == '--summary-length':
      summary_length = int(sys.argv[i+1])
      #print "summary-length: %d"%summary_length
      skip = True
    elif sys.argv[i] == '--summary-length-unit':
      key = sys.argv[i+1]
      summary_length_unit = LENGTH_UNITS[key]
      #for k in LENGTH_UNITS:
      #  if LENGTH_UNITS[k] == summary_length_unit:
      #    print "summary-length-unit: %s"%k
      skip = True
    elif sys.argv[i] == '--w2v-file':
      w2v_vector_file = sys.argv[i+1]
      skip = True
    elif sys.argv[i] == '--use-aggregate-for-clustering':
      use_aggregate_for_clustering = True
    elif sys.argv[i] == '--w2v-backend':
      w2v_backend = True
    elif sys.argv[i] == '--w2v-experiments':
      w2v_experiments = sys.argv[i+1]
      skip = True
    elif sys.argv[i] == '--quiet':
      quiet = True
    elif sys.argv[i] == '--numerical':
      output_numbers = True
    elif sys.argv[i] == '--s':
      sentences_file = sys.argv[i+1]
      skip = True
    else:
      if sys.argv[i] != "":
        files.append(sys.argv[i])
  if not quiet:
    print WELCOME_MESSAGE
    summary_length_unit_string = ""
    for s in LENGTH_UNITS:
      if summary_length_unit == LENGTH_UNITS[s]:
        summary_length_unit_string = s
	break
    print "summary_length: %s %s"%(summary_length,summary_length_unit_string)
    if input_limit:
      print "If input is longer than %d sentences, will select them randomly."%input_limit
    if sentences_file:
      print "sentences_file: %s"%sentences_file
    elif len(files) > 0:
      print "%d input files:"%len(files)
      for i in range(0, len(files)):
        print "  %s"%files[i]
  
  if doc_files:
    if not use_tfidf_similarity and not use_sentiment_similarity and not use_w2v_similarity:
      if not quiet:
        print 'Using default LinTFIDF similarity measure, since no other was provided.'
      use_tfidf_similarity = True
    summary = summarize_files(files, length=summary_length, unit=summary_length_unit, use_tfidf_similarity=use_tfidf_similarity, use_sentiment_similarity=use_sentiment_similarity, use_w2v_similarity=use_w2v_similarity, split_sentences=split_sentences, w2v_vector_file=w2v_vector_file, w2v_backend=w2v_backend, w2v_experiments=w2v_experiments, quiet=quiet, output_numbers=output_numbers, use_aggregate_for_clustering=use_aggregate_for_clustering, anaphora_resolution_simple=anaphora_resolution_simple, min_sentence_length=min_sentence_length, input_limit=input_limit)
    print summary
  else:
    summary = summarize_matrix_files(files, sentences_file, output_numbers=output_numbers, use_aggregate_for_clustering=use_aggregate_for_clustering, quiet=quiet)
    print summary
    

if  __name__ =='__main__':main()

