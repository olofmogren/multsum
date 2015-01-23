#!/usr/bin/python
# -*- coding: utf-8 -*-

import math, numpy, os.path, re, sys,time
from subprocess import call

 # @author Olof Mogren
 # 
 # Base class for submodular summarization.
 # 
 # Providing some base functions.
 #


CLUTO_SCLUSTER_EXECUTABLE = "/home/mogren/btsync/code/others_code/cluto/scluster"
CLUTO_VCLUSTER_EXECUTABLE = "/home/mogren/btsync/code/others_code/cluto/vcluster"

DEFAULT_STOPWORDS = '/home/mogren/btsync/code/others_code/ROUGE/RELEASE-1.5.5/data/smart_common_words.txt'

DEFAULT_VECTORS_FILE_PREFIX = '/tmp/submod_py_vectors.'

def getK(N):
  K = (int)(0.2 * N + 0.5)
  if K == 0: K = 1
  return K

def getClusteringByVectors(sentenceVectors, K, vectorFileName = None, docName = None, keep=False):
  clustering = list()
  if not docName:
    docName = "some_doc"
  if not vectorFileName:
    vectorFileName = DEFAULT_VECTORS_FILE_PREFIX+docName+".mat"
  outputfilename = vectorFileName+".clustering."+str(K)

  if not os.path.isfile(outputfilename):
    f = open(vectorFileName, 'w')
    #First rows, then columns:
    f.write(str(sentenceVectors.shape[0])+" "+str(sentenceVectors.shape[1])+"\n")
    for i in range(0, sentenceVectors.shape[0]):
      for j in range(0, sentenceVectors.shape[1]):
        f.write(str(sentenceVectors[i][j])+"")
        if j < sentenceVectors.shape[1]-1:
          f.write(" ")
      f.write("\n")
    f.close()

    commandList = list()
    commandList.append(CLUTO_VCLUSTER_EXECUTABLE)
    #commandList.append("-crfun=i1")
    #commandList.append("-clmethod=direct")

    #commandList.append("-clmethod=graph")
    #commandList.append("-sim=dist")

    commandList.append(vectorFileName)
    commandList.append(str(K))

    status = call(commandList)

    print(status)

    if status < 0:
      print("Error with CLUTO!")
      os.rename(vectorFileName, vectorFileName+".CLUTO_ERROR."+str(time.time()))
      exit()

    print("CLUTO status: "+str(status))
  #else:
  #  print("Using precomputed idf-clusters for "+docName)

  cf = open(outputfilename, 'r')

  index = 0
  for line in cf:
    #print("Clustering from CLUTO: "+str(index)+": "+line.replace("\n", ""))
    clustering.append(int(line))
    index += 1
  cf.close()

  if not keep:
    if os.path.isfile(vectorFileName):
      os.remove(vectorFileName)
    if os.path.isfile(outputfilename):
      os.remove(outputfilename)

  return clustering

