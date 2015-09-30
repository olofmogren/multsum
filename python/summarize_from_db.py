#!/usr/bin/python
# -*- coding: utf-8 -*-


 # Author: Olof Mogren
 #
 # email: olof.mogren@gmail.com
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



import subsum, subsum_clustering, re, os, codecs, nltk, numpy, shutil, math
#from numpy import linalg
from pymongo import MongoClient
from datetime import datetime,timedelta
from stemming.porter2 import stem
from urlparse import urlparse

SENTENCE_SPLIT_REGEX = r' *[\.\?!][\'"\)\]]* *'
AVG_DOCS_PER_CLUSTER = 3

def naive_wc_sim(str1, str2):
  list1 = nltk.word_tokenize(str1)
  list2 = nltk.word_tokenize(str2)
  count = 0
  for w1 in list1:
    stw1 = stem(w1)
    for w2 in list2:
      stw2 = stem(w2)
      if stw1 == stw2:
        count += 1
  return (1.0*count)/(1.0*min(len(list1), len(list2)))

def getTermDocumentOccurences(matrix, col):
  val = 0
  for i in range(0, matrix.shape[0]):
    if matrix[i][col] > 0:
      val += 1
  return val

def tfidfTransform(matrix):
  """ Apply TermFrequency(tf)*inverseDocumentFrequency(idf) for each matrix element.
      This evaluates how important a word is to a document in a corpus
      With a document-term matrix: matrix[x][y]
        tf[x][y] = frequency of term y in document x / frequency of all terms in document x
        idf[x][y] = log( abs(total number of documents in corpus) / abs(number of documents with term y)  )
      Note: This is not the only way to calculate tf*idf
  """

  rows,cols = matrix.shape
  documentTotal = rows

  for row in xrange(0, rows): #For each document
    wordTotal = reduce(lambda x, y: x+y, matrix[row] )
    for col in xrange(0,cols): #For each term
      #For consistency ensure all self.matrix values are floats
      matrix[row][col] = float(matrix[row][col])
      if matrix[row][col] != 0:
        termDocumentOccurences = getTermDocumentOccurences(matrix, col)
        termFrequency = matrix[row][col] / float(wordTotal)
        inverseDocumentFrequency = math.log(abs(documentTotal / float(termDocumentOccurences)))
        matrix[row][col]=termFrequency*inverseDocumentFrequency
  return matrix


def get_doc_content(doc):
  content = ''
  if 'document_title' in doc:
    content += doc['document_title']
  if 'document_preamble' in doc:
    content += " "+doc['document_preamble']
  if 'document_body' in doc:
    content += " "+doc['document_body']
  content.replace('\n', ' ')
  return content

mongo_conn = MongoClient("localhost")
mongo_db = mongo_conn.db_wordcount

#time_to = datetime.now()
time_from = datetime.now()-timedelta(days=1)

tokenizer = nltk.data.load('nltk:tokenizers/punkt/swedish.pickle')

db_ids = dict()
db_titles = dict()

file_tmpname = "/tmp/news_summary_"+str(datetime.now().strftime("%Y-%m-%d"))+".html"
file_targetname = "/home/mogren/btsync/NewsSummaries/news_summary_"+str(datetime.now().strftime("%Y-%m-%d"))+".html"

output_file = codecs.open(file_tmpname, "w", "utf-8")
output_file.write(u"<!DOCTYPE html>\n<html>\n  <head>\n    <title>News Summary for "+str(datetime.now().strftime("%Y-%m-%d"))+"</title>\n    <meta charset='utf-8'>\n  </head>\n  <body>\n\n\n");

docs = mongo_db.news_documents.find({"publication_datetime": {"$gte": time_from}, 'source_uri': {'$regex': 'nyheter'}})
titlesLists = list()
for doc in docs:
  #print('.'),
  l = list()
  l.append(doc["document_title"])
  db_ids[doc["document_title"]] = doc["_id"]
  db_titles[doc["_id"]] = doc["document_title"]
  #titlesLists.append(l)

print(str(len(db_ids))+" titles to cluster"),
noun_vocab = set()

for db_id in db_titles:
  doc = mongo_db.news_documents.find_one({'_id': db_id})
  content = get_doc_content(doc)
  tokens = nltk.word_tokenize(content)
  pos_list = nltk.pos_tag(tokens)
  for (word,pos) in pos_list:
    #noun filtering:
    if pos == "NN":
      noun_vocab.add(stem(word))

print("len, noun_vocab: "),
print(len(noun_vocab))

noun_vocab_cols = dict()
col = 0
for noun in noun_vocab:
  noun_vocab_cols[noun] = col
  col += 1

doc_vectors = numpy.zeros((len(db_titles), len(noun_vocab)))
doc_matrix_rows = dict()
doc_ids_from_matrix_rows = dict()

#third try, with noun vectors...
matrix_row = 0
for db_id in db_titles:
  doc_matrix_rows[db_id] = matrix_row
  doc_ids_from_matrix_rows[matrix_row] = db_id
  doc = mongo_db.news_documents.find_one({'_id': db_id})
  content = get_doc_content(doc)
  tokens = nltk.word_tokenize(content)
  for t in tokens:
    s = stem(t)
    if s in noun_vocab_cols:
      doc_vectors[matrix_row][noun_vocab_cols[s]] += 1
  matrix_row += 1


doc_vectors = tfidfTransform(doc_vectors)

#Sigma comes out as a list rather than a matrix
#u,sigma,vt = linalg.svd(self.matrix)

clustering = subsum_clustering.getClusteringByVectors(doc_vectors, int(len(db_titles)/AVG_DOCS_PER_CLUSTER))



clusterset = dict()

for d in clustering:
  if not clustering[d] in clusterset:
    clusterset[clustering[d]] = set()
  print("clustering "+str(clustering[d])+": adding "+str(doc_ids_from_matrix_rows[d]))
  clusterset[clustering[d]].add(doc_ids_from_matrix_rows[d])



# second try:

#to_cluster = db_ids

#while len(to_cluster) > 0:
#  centr_title, centr_id = to_cluster.popitem()
#  cluster = dict()
#  for title2 in to_cluster:
    #print(naive_wc_sim(centr_title, title2))
#    if naive_wc_sim(centr_title, title2) >= 0.25:
#      cluster[title2] = to_cluster[title2]
#  for picked_title in cluster:
#    del to_cluster[picked_title]
#  cluster[centr_title] = centr_id
#  clusters.append(cluster)


#first try:

#print("clustering... ("+str(len(titlesLists))+" titles to cluster)"),
#if len(titlesLists) == 0:
#  exit()
#clustering = subsum.get_clustering(titlesLists)
#print("done")

#inverse_clustering = list()
#for i in range(0,max(clustering)+1):
#  inverse_clustering.append(set())
#for i in range(0,len(clustering)):
#  cluster_index = clustering[i]
  #print(cluster_index),
  #print(len(inverse_clustering))
#  s = inverse_clustering[cluster_index]
#  s.add(i)
#  inverse_clustering[clustering[i]] = s

#print(inverse_clustering)

clustersetlist = clusterset.values()
clustersetlist.sort(key=lambda cluster: len(cluster), reverse=True)

for cluster in clustersetlist:
  print("***CLUSTER*** (len="+str(len(cluster))+")")
  #clusterSet = inverse_clustering[i]
  sentencesLists = list()
  titlesLists = list()
  urls = list()
  images = list()
  #for ti in clusterSet:
  for doc_id in cluster:
    #title = titlesLists[ti][0] #there are only one title/sentence in each list.
    #print("DB: title: "+title)
    print("doc no: "+str(doc_matrix_rows[doc_id])+", doc cluster: "+str(clustering[doc_matrix_rows[doc_id]]))
    docs = mongo_db.news_documents.find({"_id": doc_id})
    for doc in docs:
      print("title "+doc["document_title"])
      #print('.'),
      content = get_doc_content(doc)
      #sentencesLists.append(re.split(SENTENCE_SPLIT_REGEX, content))
      sentencesLists.append(filter(None, tokenizer.tokenize(content)))
      titlesLists.append([doc['document_title']])
      urls.append(doc['source_uri'])
      if 'image' in doc:
        images.append(doc['image'])

  print("Done with DB for this cluster. Will summarize.")
  documents = len(sentencesLists)
  sentences = 0
  for doc in sentencesLists:
    sentences += len(doc)
  print(str(documents)+' documents. '+str(sentences)+' sentences.')

  stopwords_file =  os.path.dirname(os.path.realpath(__file__))+"/swedish_stopwords.txt"
  print("stopwords_file: "+stopwords_file)

  print("(Summarized) TITLE:")
  summed_title = ''
  i = 0
  while not summed_title:
    summed_title = titlesLists[i][0]
    print('candidate title: '+summed_title)
    i += 1
  if len(titlesLists) > 1:
    summed = subsum.summarize_strings(titlesLists, stopwords_file, 1, subsum.UNIT_SENTENCES)
    if summed:
      summed_title = summed
    print('summed title: '+summed_title)
  output_file.write(u"<h1>")
  output_file.write(summed_title)#.encode('utf-8'))
  output_file.write(u"</h1>\n\n");
  if len(images) > 0:
    output_file.write(u"<p>\n");
    for u in images:
      if not 'default' in u:
        output_file.write(u"<img src=\""+u+u"\" /><br />\n");
        print(u)
    output_file.write(u"</p>\n\n\n");
  print("300 word summary of news article cluster: ")
  news_summary = ''
  if sentences > 9:
    news_summary = subsum.summarize_strings(sentencesLists, stopwords_file, 300, subsum.UNIT_WORDS)
  else:
    for l in sentencesLists:
      for s in l:
        if s:
          news_summary += s+u'\n'
  output_file.write("<p style=\"font-style: italic;\">\n");
  if len(urls) > 1:
    output_file.write(u"Automatisk sammanfattning fr&aring;n "+str(len(urls))+" olika artiklar:")
  else:
    output_file.write(u"Automatisk sammanfattning fr&aring;n "+str(len(urls))+" artikel:")
  output_file.write("</p>\n\n");
  output_file.write("<p>\n");
  news_summary = news_summary.replace("\n\n", "\n")
  news_summary = news_summary.replace("\n", "<br /><br />\n")
  output_file.write(news_summary)
  output_file.write("</p>\n\n");
  print("Källor: ")
  if len(urls) > 1:
    output_file.write(u"<p style=\"font-style: italic;\">K&auml;llor:\n");
  else:
    output_file.write(u"<p style=\"font-style: italic;\">K&auml;lla:\n");
  domain_counts = dict()
  for u in urls:
    parsed = parsed_uri = urlparse(u)
    domain = '{uri.netloc}'.format(uri=parsed)
    if domain[0:4] == 'www.':
      domain = domain[4:]
    domain_counts[domain] = domain_counts.get(domain,0)+1
    count = ''
    if domain_counts[domain] > 1:
      count = ' ('+str(domain_counts[domain])+')'
    output_file.write(u"<a href=\""+u+u"\">"+domain+count+"</a><br />\n");
    print(domain+count+': '+u)
  output_file.write(u"</p>\n\n")

  print("Done summarizing.")

output_file.write(u"  </body>\n</html>\n\n\n");
output_file.close()
shutil.copy(file_tmpname, file_targetname)


