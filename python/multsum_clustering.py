#!/usr/bin/python
# -*- coding: utf-8 -*-

import math, numpy, os.path, re, sys,time, scipy.cluster.vq, random
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


CLUTO_SCLUSTER_EXECUTABLE = "/home/mogren/sync/code/others_code/cluto/scluster"
CLUTO_VCLUSTER_EXECUTABLE = "/home/mogren/sync/code/others_code/cluto/vcluster"

DEFAULT_STOPWORDS = 'english_stopwords.txt'

DEFAULT_VECTORS_FILE_PREFIX = '/tmp/submod_py_vectors.'

MAX_RETRIES = 10

def getK(N):
  K = (int)(0.2 * N + 0.5)
  if K == 0: K = 1
  return K

def getClusteringBySimilarities(similarities, K,  docName = None, keep=False):
  return partitioning_around_medoids(similarities, K)
  try:
    print "Using DBSCAN from sklearn."
    from sklearn.cluster import DBSCAN
    #my_min_samples = (similarities.shape[0]/K)
    max_min_samples = 10
    min_eps = 0.5
    my_min_samples = 2
    done = False
    my_eps = 0.88
    while not done:
      clustering = DBSCAN(min_samples=my_min_samples, eps=my_eps, metric='precomputed').fit_predict(1-similarities)
      num_outliers = len([itm for itm in clustering if itm == -1])
      num_clusters = max(clustering)
      if num_outliers > (similarities.shape[0]/2):
        my_eps -= .01
        print "Too many outliers! (%d). Setting eps to %f."%(num_outliers, my_eps)
        done = False
        if my_eps < min_eps:
          done = True
      elif num_clusters > K:
        my_min_samples += 1
        print "Too many clusters! (%d). Setting my min_samples to %d."%(num_clusters, my_min_samples)
        done = False
        if my_min_samples > max_min_samples:
          done = True
      elif num_clusters < 2:
        my_eps -= .01
        print "Too few clusters! (%d). Setting my eps to %f."%(num_clusters, my_eps)
        done = False
        if my_eps < min_eps:
          done = True
      else:
        done = True
    print "DBSCAN done. num_clusters %d"%(max(clustering))
    print "Assigning outliers to their own clusters of size one."
    cluster_number = max(clustering)+1
    for i in range(0, len(clustering)):
      if clustering[i] == -1:
        clustering[i] = cluster_number
        cluster_number += 1
    return clustering
  except Exception,e:
    print e
    print "Failed to import DBSCAN from sklearn.cluster"
    if CLUTO_SCLUSTER_EXECUTABLE and os.path.isfile(CLUTO_SCLUSTER_EXECUTABLE):
      return getClusteringCluto(similarities, K,  docName, keep, simclustering=True)
    else:
      print("Did not find cluto binary (looked in %s). Will try to cluster using scipy."%(CLUTO_VCLUSTER_EXECUTABLE))
      exit()
def getClusteringByVectors(sentenceVectors, K, matrixFileName = None, docName = None, keep=False):
  if CLUTO_VCLUSTER_EXECUTABLE and os.path.isfile(CLUTO_VCLUSTER_EXECUTABLE):
    return getClusteringCluto(sentenceVectors, K, matrixFileName, docName, keep, simclustering=False)
  else:
    print("Did not find cluto binary (looked in %s). Will try to cluster using scipy."%(CLUTO_VCLUSTER_EXECUTABLE))
    return getClusteringByVectorsScipy(sentenceVectors, K)

def getClusteringCluto(matrix, K, matrixFileName = None, docName = None, keep=False, simclustering=False):
  clustering = list()
  if not docName:
    docName = "some_doc"
  if not matrixFileName:
    matrixFileName = DEFAULT_VECTORS_FILE_PREFIX+docName+".mat"
  outputfilename = matrixFileName+".clustering."+str(K)

  if not os.path.isfile(outputfilename):
    f = open(matrixFileName, 'w')
    #First rows, then columns:
    f.write(str(matrix.shape[0])+" "+str(matrix.shape[1])+"\n")
    for i in range(0, matrix.shape[0]):
      for j in range(0, matrix.shape[1]):
        f.write(str(matrix[i][j])+"")
        if j < matrix.shape[1]-1:
          f.write(" ")
      f.write("\n")
    f.close()

    commandList = list()
    if simclustering:
      commandList.append(CLUTO_SCLUSTER_EXECUTABLE)
    else:
      commandList.append(CLUTO_VCLUSTER_EXECUTABLE)
    #commandList.append("-crfun=i1")
    #commandList.append("-clmethod=direct")

    #commandList.append("-clmethod=graph")
    #commandList.append("-sim=dist")

      commandList.append(matrixFileName)
      commandList.append(str(K))

      status = call(commandList)

      print(status)

      if status < 0:
        print("Error with CLUTO!")
        os.rename(matrixFileName, matrixFileName+".CLUTO_ERROR."+str(time.time()))
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
      if os.path.isfile(matrixFileName):
        os.remove(matrixFileName)
      if os.path.isfile(outputfilename):
        os.remove(outputfilename)

    return clustering

  def getClusteringBySimilaritiesScipy(similarities, K):
    retries = 0
    while retries < MAX_RETRIES:
      try:
        (centroid, label) = scipy.cluster.vq.kmeans2(similarities, K, minit='points')#, iter=10, thresh=1e-05, missing='warn', check_finite=True)
      except:
        print("Some error in clustering. Retrying max %d times. %d"%(MAX_RETRIES, retries))
    return label

def getClusteringByVectorsScipy(sentenceVectors, K):
  retries = 0
  while retries < MAX_RETRIES:
    try:
      (centroid, label) = scipy.cluster.vq.kmeans2(sentenceVectors, K, minit='points')#, iter=10, thresh=1e-05, missing='warn', check_finite=True)
    except:
      print("Some error in clustering. Retrying max %d times. %d"%(MAX_RETRIES, retries))
  return label

'''
  Initialize: randomly select[citation needed] (without replacement) k of the n data points as the medoids
  Associate each data point to the closest medoid.
  While the cost of the configuration decreases:
    For each medoid m, for each non-medoid data point o:
      Swap m and o, recompute the cost
      If the total cost of the configuration increased in the previous step, undo the swap

'''
def partitioning_around_medoids(similarities, K):
  num_sentences = similarities.shape[0]
  sentences = range(0,num_sentences)
  clustering = [0]*num_sentences
  candidate_clustering = [0]*num_sentences
  candidate_medoids = [0]*K
  #Initialization:
  print "K: %d"%K
  medoids = random.sample(sentences, K)
  print "medoids",
  for m in medoids:
    print m,
  for s in sentences:
    if s in medoids:
      clustering[s] = medoids.index(s)
    else:
      max_sim = 0.0
      closest_medoid = -1
      for m in medoids:
        if similarities[s,m] > max_sim:
          closest_medoid = medoids.index(m)
          max_sim = similarities[s,m]
      clustering[s] = closest_medoid
  #Initialization done.
  current_utility = clustering_utility(similarities, medoids, clustering)
  print "first utility: %f"%current_utility
  improving = True
  iterations = 0
  while improving:
    improving = False
    iterations += 1
    print "current utility: %f. iteration %d."%(current_utility, iterations)
    for m in medoids:
      #if improving:
        # we have made a swap, so need
      #print "medoid: %d"%m
      for s in sentences:
        if s in medoids:
          continue
        #print "non-medoid sentence: %d"%s
        #max_sim = 0.0
        #closest_medoid = -1
        #for m2 in medoids:
        #  if similarities[m,m2] > max_sim:
        #    closest_medoid = medoids.index(m2)
        #    max_sim = similarities[m,m2]
        #candidate_clustering_for_m = closest_medoid
        candidate_medoids = list()
        for m2 in medoids:
          if m2 == m:
            candidate_medoids.append(s)
          else:
            candidate_medoids.append(m2)
        for s2 in sentences:
          if s2 in medoids:
            candidate_clustering[s2] = medoids.index(s2)
          else:
            max_sim = 0.0
            closest_medoid = -1
            for m2 in candidate_medoids:
              if similarities[s2,m2] > max_sim:
                closest_medoid = candidate_medoids.index(m2)
                max_sim = similarities[s2,m2]
            candidate_clustering[s2] = closest_medoid
        candidate_utility = clustering_utility(similarities, candidate_medoids, candidate_clustering)
        #print "candidate utility: %f. iteration %d."%(candidate_utility, iterations)
        if candidate_utility > current_utility:
          improving = True
          current_utility = candidate_utility
          clustering = [i for i in candidate_clustering]
          medoids = [i for i in candidate_medoids]
          #medoid_id = medoids.index(m)
          #medoids[medoid_id] = s
          #clustering[s] = medoid_id
          #clustering[m] = candidate_clustering_for_m
          # pick next centroid. This one was swapped now.
          break
  print "PMA done. iterations: %d"%iterations
  for i in clustering:
    print i,
  return clustering

'''


Select initial medoids
Iterate while the cost (sum of distances of points to their medoid) decreases:
  In each cluster, make the point that minimizes the sum of distances within the cluster the medoid
  Reassign each point to the cluster defined by the closest medoid determined in the previous step.

'''
def voronoi_iteration(similarities, K):
  num_sentences = similarities.shape[0]
  sentences = range(0,num_sentences)
  clustering = [0]*num_sentences
  candidate_clustering = [0]*num_sentences
  candidate_medoids = [0]*K
  #Initialization:
  print "K: %d"%K
  medoids = random.sample(sentences, K)
  print "medoids",
  for m in medoids:
    print m,
  for s in sentences:
    if s in medoids:
      clustering[s] = medoids.index(s)
    else:
      max_sim = 0.0
      closest_medoid = -1
      for m in medoids:
        if similarities[s,m] > max_sim:
          closest_medoid = medoids.index(m)
          max_sim = similarities[s,m]
      clustering[s] = closest_medoid
  #Initialization done.
  current_utility = clustering_utility(similarities, medoids, clustering)
  print "first utility: %f"%current_utility
  improving = True
  iterations = 0
  while improving:
    improving = False
    iterations += 1
    print "current utility: %f. iteration %d."%(current_utility, iterations)
    for c in xrange(0,medoids):
      raise("TIME FOR SLEEP!")
      for s in sentences:
        if s in medoids:
          continue
        #print "non-medoid sentence: %d"%s
        #max_sim = 0.0
        #closest_medoid = -1
        #for m2 in medoids:
        #  if similarities[m,m2] > max_sim:
        #    closest_medoid = medoids.index(m2)
        #    max_sim = similarities[m,m2]
        #candidate_clustering_for_m = closest_medoid
        candidate_medoids = list()
        for m2 in medoids:
          if m2 == m:
            candidate_medoids.append(s)
          else:
            candidate_medoids.append(m2)
        for s2 in sentences:
          if s2 in medoids:
            candidate_clustering[s2] = medoids.index(s2)
          else:
            max_sim = 0.0
            closest_medoid = -1
            for m2 in candidate_medoids:
              if similarities[s2,m2] > max_sim:
                closest_medoid = candidate_medoids.index(m2)
                max_sim = similarities[s2,m2]
            candidate_clustering[s2] = closest_medoid
        candidate_utility = clustering_utility(similarities, candidate_medoids, candidate_clustering)
        #print "candidate utility: %f. iteration %d."%(candidate_utility, iterations)
        if candidate_utility > current_utility:
          improving = True
          current_utility = candidate_utility
          clustering = [i for i in candidate_clustering]
          medoids = [i for i in candidate_medoids]
          #medoid_id = medoids.index(m)
          #medoids[medoid_id] = s
          #clustering[s] = medoid_id
          #clustering[m] = candidate_clustering_for_m
          # pick next centroid. This one was swapped now.
          break
  print "PMA done. iterations: %d"%iterations
  for i in clustering:
    print i,
  return clustering



def clustering_utility(similarities, medoids, clustering):
  cluster_utility = [0.0]*len(medoids)
  for c in range(0,len(medoids)):
    medoid = medoids[c]
    for s in range(0,len(clustering)):
      cluster = clustering[s]
      if c == cluster:
        cluster_utility[cluster] = cluster_utility[cluster]+similarities[medoid][s]
  return sum(cluster_utility)

def clustering_utility_with_swaps(similarities, medoids, clustering, medoid_to_swap=None, sentence_to_swap=None, candidate_clustering_for_medoid=None):
  cluster_utility = [0.0]*len(medoids)
  for c in range(0,len(medoids)):
    medoid = medoids[c]
    if medoid == medoid_to_swap:
      medoid = sentence_to_swap
    for s in range(0,len(clustering)):
      cluster = get_cluster_with_swap(clustering, medoid_to_swap, sentence_to_swap, candidate_clustering_for_medoid, s)
      if c == cluster:
        cluster_utility[cluster] = cluster_utility[cluster]+similarities[medoid][s]
  return sum(cluster_utility)
  #punishment = 0.0
  #for i in range(0,len(medoids)):
  #  for j in range(i+1,len(medoids)):
  #    medoid1 = medoids[i]
  #    medoid2 = medoids[j]
  #    if medoid1 == medoid_to_swap:
  #      medoid1 = sentence_to_swap
  #    elif medoid2 == medoid_to_swap:
  #      medoid2 = sentence_to_swap
  #    punishment += similarities[medoid1][medoid2]
  #return sum(cluster_utility) - punishment

def get_cluster_with_swap(clustering, medoid_to_swap, sentence_to_swap, candidate_clustering_for_medoid, s):
  if medoid_to_swap == None or sentence_to_swap == None or candidate_clustering_for_medoid == None:
    return clustering[s]
  if s == medoid_to_swap:
    return clustering[sentence_to_swap]
  elif s == sentence_to_swap:
    return candidate_clustering_for_medoid
  else:
    return clustering[s]

