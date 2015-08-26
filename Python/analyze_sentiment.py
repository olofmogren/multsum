#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys, re, math, numpy
from sets import Set

f = open(os.path.dirname(__file__)+"/emotion_words_positive.txt")
#positive_emotions = Set(f.readlines())
positive_emotions = set()
for pos in f.readlines():
  #print pos
  positive_emotions.add(pos.replace("\n", ""))
f = open(os.path.dirname(__file__)+"/emotion_words_negative.txt")
negative_emotions = set()
for neg in f.readlines():
  #print neg
  negative_emotions.add(neg.replace("\n", ""))

def analyze_sentiment(sentences):
  emo_vectors = []
  for s in sentences:
    #print s
    positive_count = 0
    negative_count = 0
    positive_frac = 0.0
    negative_frac = 0.0
    #words = s.split()
    words = re.findall(r"[\w']+", s)
    if len(words) > 0:
      for w in words:
        #"print w
        if w.lower() in positive_emotions:
          #print "positive! "+w
          positive_count += 1
        elif w.lower() in negative_emotions:
          #print "negative! "+w
          negative_count += 1
      positive_frac = float(positive_count) / len(words)
      negative_frac = float(negative_count) / len(words)
    emo_vec = [positive_frac, negative_frac]
    emo_vectors.append(emo_vec)
  # cosinesim is btw -1 and 1.
  # use:
  # 1+cosinesimilarity(vec1,vec2)/2 

  positive_matrix = numpy.zeros((len(sentences), len(sentences)))
  negative_matrix = numpy.zeros((len(sentences), len(sentences)))

  min_simpos = 1.0
  max_simpos = 0.0
  min_simneg = 1.0
  max_simneg = 0.0
  #print "Number of lines: "+str(len(emo_vectors))
  for normalize in [1,0]:
    for i in range(len(emo_vectors)):
      for j in range(len(emo_vectors)):
        simpos = 1-abs(emo_vectors[i][0]-emo_vectors[j][0])
        simneg = 1-abs(emo_vectors[i][1]-emo_vectors[j][1])
        sim = simpos
        if normalize == 1:
          if simpos > max_simpos:
            max_simpos = simpos
          if simpos < min_simpos:
            min_simpos = simpos
          if simneg > max_simneg:
            max_simneg = simneg
          if simneg < min_simneg:
            min_simneg = simneg
        else:
          normalized_simpos = 0.0
          normalized_simneg = 0.0
          if len(sentences[i].replace('\n', '').strip()) > 0 and len(sentences[j].replace('\n', '').strip()) > 0:
            if max_simpos-min_simpos > 0.0:
              normalized_simpos = (simpos-min_simpos)/(max_simpos-min_simpos)
            if max_simneg-min_simneg > 0.0:
              normalized_simneg = (simneg-min_simneg)/(max_simneg-min_simneg)

          positive_matrix[i][j] = normalized_simpos
          negative_matrix[i][j] = normalized_simneg

  return (positive_matrix, negative_matrix)

