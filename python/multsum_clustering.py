#!/usr/bin/python
# -*- coding: utf-8 -*-

import math, numpy, os.path, re, sys,time, scipy.cluster.vq
from subprocess import call

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


#CLUTO_SCLUSTER_EXECUTABLE = "/home/mogren/sync/code/others_code/cluto/scluster"

#CLUTO_VCLUSTER_EXECUTABLE = "/home/mogren/sync/code/others_code/cluto/vcluster"
CLUTO_VCLUSTER_EXECUTABLE = None

DEFAULT_STOPWORDS = 'english_stopwords.txt'

DEFAULT_VECTORS_FILE_PREFIX = '/tmp/submod_py_vectors.'

def getK(N):
  K = (int)(0.2 * N + 0.5)
  if K == 0: K = 1
  return K

def getClusteringByVectors(sentenceVectors, K, vectorFileName = None, docName = None, keep=False):
  if CLUTO_VCLUSTER_EXECUTABLE and os.path.isfile(CLUTO_VCLUSTER_EXECUTABLE):
    return getClusteringByVectorsCluto(sentenceVectors, K, vectorFileName, docName, keep)
  else:
    print("Did not find cluto binary (looked in %s). Will try to cluster using scipy."%(CLUTO_VCLUSTER_EXECUTABLE))
    return getClusteringByVectorsScipy(sentenceVectors, K, vectorFileName, docName, keep)

def getClusteringByVectorsCluto(sentenceVectors, K, vectorFileName = None, docName = None, keep=False):
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

def getClusteringByVectorsScipy(sentenceVectors, K, vectorFileName = None, docName = None, keep=False):
  (centroid, label) = scipy.cluster.vq.kmeans2(sentenceVectors, K, minit='points')#, iter=10, thresh=1e-05, missing='warn', check_finite=True)
  return label

