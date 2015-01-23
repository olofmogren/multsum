#!/usr/bin/python
# -*- coding: utf-8 -*-

import math, numpy, os.path, re, sys,time
from stemming.porter2 import stem
from subsum_clustering import *

 # @author Olof Mogren
 #
 # email: olof.mogren@gmail.com
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

REGEX_SPACE         = "\\W+"
REGEX_SPACE_COMMA   = "\\s+|,"
REGEX_NONWORD       = "[^\\p{L}\\p{Nd}]+"

DEFAULT_STOPWORDS = '/home/mogren/btsync/code/others_code/ROUGE/RELEASE-1.5.5/data/smart_common_words.txt'

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

  aggregateSim = zeros((ms.shape[0],ms.shape[1]))

  for m in ms:
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

def summaryIsTooShort(selected, sentencesLists, lengthUnit, summarySize):
  if lengthUnit == UNIT_CHARACTERS:
    return charactersLength(selected, sentencesLists) < summarySize
  elif lengthUnit == UNIT_WORDS:
    return wordsLength(selected, sentencesLists) < summarySize
  else:
    return len(selected) < summarySize
def summaryIsTooLong(selected, sentencesLists, lengthUnit, summarySize):
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

  K = getK(count_sentences(sentencesLists))
  clustering = getClusteringByVectors(sentenceVectors, K, idfVectorFileName, docName)

  while summaryIsTooShort(selected, sentencesLists, lengthUnit, summarySize):
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
    if not summaryIsTooLong(singleton, sentencesLists, lengthUnit, summarySize):
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


def summarize_strings(sentencesLists, stopwords=DEFAULT_STOPWORDS, length=300, unit=UNIT_WORDS):
  sentsims = get_def_sentsims(sentencesLists, stopwords, None)

  matrices = list()
  matrices.append(sentsims["tfidf_cosine"])

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
  for i in summary_list:
    return_string += get_sentence_index(i, sentencesLists)+'\n'
    print(get_sentence_index(i, sentencesLists))
  return return_string
 

def summarize_files(document_names, length=300, unit=UNIT_WORDS):
  sentencesLists = list()
  for filename in document_names:
    f = open(filename, 'r')
    sentences = list()
    for line in f:
      if line:
        sentences.append(line)
    sentencesLists.append(sentences)

  return summarize_strings(sentencesLists, length, unit)

def get_clustering(sentencesLists, stopwords=DEFAULT_STOPWORDS):
  sentsims = get_def_sentsims(sentencesLists, stopwords, None)
  K = getK(sentsims["tfidf_cosine"].shape[0])
  clustering = getClusteringByVectors(sentsims["idf_vectors"], K, None, "summarization_doc")
  return clustering


def main():
  l = list()
  for i in range(1,len(sys.argv)):
    l.append(sys.argv[i])
  summarize_files(l)

if  __name__ =='__main__':main()

