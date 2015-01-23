package se.chalmers.mogren.submodularsummarization;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import se.chalmers.mogren.submodularsummarization.util.PorterStemmer;

/**
 * 
 * $author Olof Mogren, mogren@chalmers.se
 * 
 * Referred to as "beta" in the evaluator. Don't remember exactly why. IDF-vectors clustering?
 * This version includes alternative ways of aggregating measures.
 * 
 * This is an implementation of Submodular Optimization for Summarization,
 * due to Lin&Bilmes.
 * 
 * The approach is described in:
 * Lin, Hui; Bilmes, Jeff - 2011 - A Class of Submodular Functions for Document Summarization - ACL - 2011.
 *
 * This implementation is released under the GNU GPL.
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

public class SubmodularSummarization2 extends SubmodularSummarization
{
  private static final String USAGE_STRING = "Usage: <command> --matrix-file <filename1> [--matrix-file <filename2> [--matrix-file <filename3> [...]]] \n" +
      "--document-file file-or-directory \n" +
      "--collection-for-idfs directory \n" +
      "--stopwords-file filename \n" +
      "--word-clusters-file filename \n" +
      "[--nb-sentences summarysize] \n" +
      "[--nb-words summarysize] \n" +
      "[--nb-characters summarysize] \n" +
      "[--lambda lambda] \n" +
      "[--disable-bigrams] \n" +
      "[--print-as-text] \n" +
      "[--flag MOD_COST_WEIGHT|MOD_SINGLETON|MOD_STRICT_LENGTH ] \n" +
      "(Either at least one matrix file or a document file must be present.)\n" +
      "(If document file is a directory, I will try to summarize all documents within.)\n";



  private static final double A                    = 5.0;
  //  private static final double ALPHA                = 0.75f; //LIn-bilmes: alpha = a/N
  private static final double DEFAULT_LAMBDA       = 6.0;

  private static final double LINBILMES_CONSTANT_R = 0.3f;
  LinkedList<Integer> selectedList;


  public static void main(String[] args)
  {
    SubmodularSummarization2 summary = null;
    double lambda = DEFAULT_LAMBDA;
    int summarySize = 2;
    LengthUnit summarySizeUnit = LengthUnit.SENTENCES;
    String stopwordsFilename = null;
    String documentFileName = null;
    String wordClusterFileName = null;
    String idfCollection = null;
    HashMap<String, Double> idfs = null;
    HashSet<Flags> flags = new HashSet<Flags>();
    boolean printAsText = false;

    //DEFAULT:
    flags.add(Flags.MOD_BIGRAMS);

    if (args.length <= 0)
    {
      System.out.println(USAGE_STRING);
      System.exit(1);
    }

    //New usage mode
    ArrayList<String> matrixFileNames = new ArrayList<String>();
    for(int i=0; i<args.length; i++)
    {
      if(args[i].equals("--matrix-file"))
      {
        matrixFileNames.add(args[++i]);
      }
      if(args[i].equals("--collection-for-idfs"))
      {
        idfCollection = args[++i];
      }
      else if(args[i].equals("--document-file"))
      {
        documentFileName = args[++i];
      }
      else if(args[i].equals("--stopwords-file"))
      {
        stopwordsFilename = args[++i];
      }
      else if(args[i].equals("--word-clusters-file"))
      {
        wordClusterFileName = args[++i];
      }
      else if(args[i].equals("--nb-sentences"))
      {
        summarySize = Integer.parseInt(args[++i]);
        summarySizeUnit = LengthUnit.SENTENCES;
      }
      else if(args[i].equals("--nb-words"))
      {
        summarySize = Integer.parseInt(args[++i]);
        summarySizeUnit = LengthUnit.WORDS;
      }
      else if(args[i].equals("--nb-characters"))
      {
        summarySize = Integer.parseInt(args[++i]);
        summarySizeUnit = LengthUnit.CHARACTERS;
      }
      else if(args[i].equals("--lambda"))
      {
        lambda = (double)Integer.parseInt(args[++i]);
      }
      else if(args[i].equals("--disable-bigrams"))
      {
        flags.remove(Flags.MOD_BIGRAMS);
      }
      else if(args[i].equals("--flag"))
      {
        flags.add(Flags.valueOf(args[++i]));
      }
      else if(args[i].equals("--print-as-text"))
      {
        printAsText = true;
      }
      else
      {
        System.out.println("Unknown argument \""+args[i]+"\"\n");
        System.out.println(USAGE_STRING);
        System.exit(0);
      }
    }

    List<String> sentences = null;
    if(documentFileName != null)
    {
      sentences =  getSentencesList(documentFileName);
    }
    //for(String s: sentences)
    //  System.out.println(s);

    if(idfCollection != null)
    {
      idfs = getIdfsDocCluster(idfCollection, stopwordsFilename, wordClusterFileName);
      TreeSet<String> tree = new TreeSet<String>(idfs.keySet());
      try
      {
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(System.getProperty("user.home")+"/tmp/idfs.txt")));
        for(String term: tree)
        {
          bw.write(term+": "+idfs.get(term)+"\n");
        }
        bw.close();
      }
      catch(IOException e)
      {
        e.printStackTrace();
      }
    }

    if(matrixFileNames.size() > 0)
    {
      if(sentences == null)
        System.err.println("WARNING! sentences list is null!");

      HashMap<String, Double> nullIdfs = null;
      summary = new SubmodularSummarization2(matrixFileNames, null, sentences, null, null, null, lambda, summarySize, summarySizeUnit, flags, nullIdfs, null);
    }
    else
    {
      summary = new SubmodularSummarization2(null, null, null, stopwordsFilename, documentFileName, wordClusterFileName, lambda, summarySize, summarySizeUnit, flags, idfs, null);
    }

    if(printAsText)
      System.out.println(summary.getSummaryAsText(sentences));

    System.out.println(summary);

  }
  public SubmodularSummarization2(List<String> matrixFileNames,
      String weightFile,
      List<String> sentences,
      String stopwordsFilename,
      String sentencesFileName,
      String wordClusterFileName,
      Double lambda,
      int summarySize,
      LengthUnit lengthUnit,
      Collection<Flags> flags,
      String pathToDocumentsForIDF,
      String idfVectorsFileName)
  {
    this(matrixFileNames,
        weightFile,
        sentences,
        stopwordsFilename,
        sentencesFileName,
        wordClusterFileName,
        lambda,
        summarySize,
        lengthUnit,
        flags,
        getIdfsDocClusterOrSentences(pathToDocumentsForIDF, sentences, stopwordsFilename, wordClusterFileName),
        idfVectorsFileName);
  }
  /*
   * Constructor for SubmodularSummarization2
   * 
   * At least one of matrixFileNames or sentences or sentencesFileName need to be present.  
   * 
   * @param matrixFileNames     Filenames for matrices. The matrices
   *                            should contain similarity information between sentences.
   *                            If this is provided, sentences or sentencesFileName
   *                            are not used for sentence similarity computation
   *                            (but if either of them are available, they are used for
   *                            clustering sentences).
   * @param weightFile          If matrixFileNames are provided, weightFile can be provided
   *                            to let SubmodularOptimization use weighted averages for
   *                            similarity scores. It's size should be a (any) multiple of
   *                            the number of matrices M. The first M entries in this array
   *                            will be used as weights for choosing the first sentence to the
   *                            summary. Then the next M entries for the next sentence and so on.
   * @param sentences           A list of sentences to summarize. If this is provided,
   *                            sentencesFileName is ignored. The sentences are used for computed
   *                            (default) sentence similarities if matrixFileNames are not
   *                            provided, and even if there are matrices, the sentences are
   *                            used for the clustering.
   * @param stopwordsFileName   A file containing stopwords. One word per line.
   *                            This list of stopwords is used when computing
   *                            (default) sentence similarities.
   * @param sentencesFileName   A file containing sentences to summarize. One sentence per line.
   *                            The sentences are used for computing
   *                            (default) sentence similarities if matrixFileNames are not
   *                            provided, and even if there are matrices, the sentences are
   *                            used for the clustering.
   *                            If sentencesFileName is a directory, the program will
   *                            scan this directory for text files to use as input.
   * @param wordclusterFileName A file containing word clusters. Words in the input text
   *                            will be replaced by their corresponding cluster id:s
   *                            before summarization.
   * @param lambda              Lambda, see paper by Lin&Bilmes.
   * @param summarySize         Summary size
   * @param lengthUnit          Length unit
   * @param flags               Flags that can change the default behaviour of the program.
   * @param idfs                A map of idf scores for this text. If there is any term missing,
   *                            it will get a default idf score of Math.log10(10.0/1.0) (= 1.0).
   *                            If idfs are not provided at all (null), it will be computed as
   *                            getIdfsFromDocCollection(sentencesLists, stopwordsFilename, wordClusterFileName)
   *
   */
  public SubmodularSummarization2(List<String> matrixFileNames,
      String matrixWeightFile,
      List<String> sentences,
      String stopwordsFilename,
      String sentencesFileName,
      String wordClusterFileName,
      Double lambda,
      int summarySize,
      LengthUnit lengthUnit,
      Collection<Flags> flags,
      HashMap<String, Double> idfs,
      String idfVectorsFileName)
  {
    DocMatrices[] matrices = null;

    if(lambda == null)
      lambda = DEFAULT_LAMBDA;

    if(matrixFileNames != null)
    {
      ArrayList<double[][]> sims = new ArrayList<double[][]>();
      for(String f: matrixFileNames)
      {
        sims.add(getSimilaritiesFromFile(f));
      }

      matrices = new DocMatrices[sims.size()];
      for(int i = 0; i < sims.size(); i++)
      {
        matrices[i] = new DocMatrices(sims.get(i), getDissimilarities(sims.get(i)));
      }
    }

    double[] weights = null;
    if(matrixWeightFile != null)
      weights = getWeightsFromFile(matrixWeightFile);

    List<List<String>> sentencesLists = null;
    if(sentences != null)
    {
      sentencesLists = new ArrayList<List<String>>();
      sentencesLists.add(sentences);
    }
    else if(sentencesFileName != null)
    {
      sentences = new ArrayList<String>();
      sentencesLists = getSentencesLists(sentencesFileName);
      for(List<String> list: sentencesLists)
        sentences.addAll(list);
    }

    double[][] sentenceVectorsForClustering = null;
    if(sentencesLists != null)
    {
      DocMatrices[] sentenceMeasures = new DocMatrices[1];
      sentenceMeasures[0] = getDefaultSentenceSimilarities(stopwordsFilename, sentencesLists, wordClusterFileName, flags, idfs);
      sentenceVectorsForClustering = sentenceMeasures[0].sentenceVectors;
      if(matrices == null)
        matrices = sentenceMeasures;
    }
    //Cosine similarity should be between 0 and 1, if vectors are in positive space!

    if(matrixFileNames == null && flags.contains(Flags.OPTION_SAVE_MATRICES))
    {
      try
      {
        saveLinTFIDFMatrix(matrices[0].similarities, new File(sentencesFileName).getParentFile().getName());
      }
      catch(IOException e)
      {
        e.printStackTrace();
      }
    }

    selectSentences(summarySize, matrices, lambda, sentences, lengthUnit, flags, weights, sentenceVectorsForClustering, idfVectorsFileName, new File(sentencesFileName).getParentFile().getName());
  }

  private static List<String> getSentencesList(String documentFileName)
  {
    List<List<String>> sentencesLists = getSentencesLists(documentFileName);
    List<String> sentences = new ArrayList<String>();
    for(List<String> list: sentencesLists)
      sentences.addAll(list);
    return sentences;
  }

  private static List<List<String>> getSentencesLists(String documentFileName)
  {
    List<List<String>> sentencesLists = new ArrayList<List<String>>();
    ArrayList<File> files = new ArrayList<File>();

    File documentFile = new File(documentFileName);
    if(documentFile.isDirectory())
      for(File f: documentFile.listFiles())
        files.add(f);
    else
      files.add(documentFile);
    try
    {
      for(File f: files)
      {
        ArrayList<String> document = new ArrayList<String>();
        BufferedReader reader = new BufferedReader(new FileReader(f));
        String line = reader.readLine();
        while (line != null)
        {
          document.add(line);
          line = reader.readLine();
        }
        reader.close();
        sentencesLists.add(document);
      }
    }
    catch(IOException e)
    {
      e.printStackTrace();
      System.exit(-1);
    }
    return sentencesLists;
  }

  private DocMatrices getDefaultSentenceSimilarities(String stopwordsFilename, List<List<String>> sentencesLists, String wordClusterFileName, Collection<Flags> flags, HashMap<String, Double> idfs)
  {
    //Some initializations:
    double[][] sentenceTFIDFCosineSim = null;
    double[][] sentenceDissimilarities = null;
    double[][] sentenceTFIDFEuclidean = null;
    double[][] sentenceTFIDFVectors = null;
    double[][] sentenceIDFVectors = null;

    //System.out.println ("Using internal TFIDF-based sentence similaritites.");

    HashSet<String> stopwords = getStopwords(stopwordsFilename);
    HashMap<String, String> wordClusterId = getWordClusters(wordClusterFileName);

    List<Map<String, Integer>> sentencesBags = getSentencesAsBags(stopwords, sentencesLists, wordClusterId, flags, idfs);

    List<String> vocabulary = new ArrayList<String>();
    for(Map<String, Integer> sentence: sentencesBags)
      for(String term: sentence.keySet())
        vocabulary.add(term);
    Collections.sort(vocabulary);

    HashMap<String, Integer> vocabularyIndices = new HashMap<String, Integer>();
    for(int i = 0; i < vocabulary.size(); i++)
      vocabularyIndices.put(vocabulary.get(i), i);

    sentenceTFIDFVectors = new double[sentencesBags.size()][vocabulary.size()];
    sentenceIDFVectors = new double[sentencesBags.size()][vocabulary.size()];
    for(int i = 0; i < sentenceTFIDFVectors.length; i++)
      for(int j = 0; j < sentenceTFIDFVectors[0].length; j++)
      {
        sentenceTFIDFVectors[i][j] = 0.0;
        sentenceIDFVectors[i][j] = 0.0;
      }

    if(idfs == null)
    {
      // The following is what lin-bilmes did, if sentencesLists contains each document in a cluster.
      idfs = getIdfsFromDocCollection(sentencesLists, stopwordsFilename, wordClusterFileName);
    }

    /* Denominators for the cosine similarity computation: */
    double[] tfidfden = new double[sentencesBags.size()];
    double[] idfden = new double[sentencesBags.size()];
    for (int i = 0; i < sentencesBags.size(); i++)
    {
      tfidfden [i] = idfden [i] = 0.0;
      for(String term: sentencesBags.get(i).keySet())
      {
        Integer tf = sentencesBags.get(i).get(term);
        Double idf = idfs.get(term);
        if(idf==null)
        {
          //Ugly hack. Because of some mismatch in sentence splitting on DUC, you sometimes get idfs not found for some bigrams. Will treat as if present in one document.
          idf = 1.0;
          idfs.put(term, idf);
          System.err.println("No idf for "+term+"! ");
        }
        if(tf==null)
          System.err.println("No tf for "+term+"! STRANGE!");
        //Double tfidf = ((1+Math.log10(tf))*idf); //manning coursera nlp-course
        Double tfidf = tf.doubleValue()*idf; //lin-bilmes paper.

        sentenceTFIDFVectors[i][vocabularyIndices.get(term)] = tfidf;
        sentenceIDFVectors[i][vocabularyIndices.get(term)] = idf;

        tfidfden[i] += tfidf * tfidf;
        idfden[i] += idf * idf;
      }
      tfidfden[i] = Math.sqrt(tfidfden [i]);
      idfden[i] = Math.sqrt(idfden [i]);
    }


    /* Numerators for the cosine similarity computation: */
    double[][] tfidfsim = new double[sentencesBags.size()][sentencesBags.size()];
    double[][] idfdist = new double[sentencesBags.size()][sentencesBags.size()];
    sentenceTFIDFEuclidean = new double[sentencesBags.size()][sentencesBags.size()];

    for (int i = 0; i != sentencesBags.size(); ++i)
    {
      tfidfsim [i][i] = 1.0;
      idfdist [i][i] = 0.0;
      for (int j = i + 1; j != sentencesBags.size(); ++j)
      {
        double euclideanSum = 0.0;
        double tfidfnum = 0.0, idfnum = 0.0;
        for(String term: sentencesBags.get(i).keySet())
        {
          Integer tf_i = sentencesBags.get(i).get(term);
          Integer tf_j = sentencesBags.get(j).get(term);
          if(tf_i == null) tf_i = 0; if( tf_j == null) tf_j = 0;
          Double idf = idfs.get(term);
          if(idf==null)
          {
            //Ugly hack. Because of some mismatch in sentence splitting on DUC, you sometimes get idfs not found for some bigrams. Will treat as if present in one document.
            idf = 1.0;
            idfs.put(term, idf);
            System.err.println("No idf for "+term+"! ");
          }

          euclideanSum += Math.pow(tf_i*idf-tf_j*idf, 2);

          //Double tfidf =  ((1+Math.log10(tf))*idf); //manning coursera nlp-course
          Double tfidf_i = ((double)tf_i)*idf; //lin-bilmes paper.
          Double tfidf_j = ((double)tf_j)*idf; //lin-bilmes paper.
          tfidfnum += tfidf_i * tfidf_j;
          idfnum += idf * idf;
        }
        tfidfsim[i][j] = tfidfsim[j][i] = (tfidfden[i]==0 || tfidfden[j]==0) ? 0.0 : tfidfnum / (tfidfden[i] * tfidfden[j]);
        idfdist[i][j] = idfdist[j][i] = (idfden[i]==0 || idfden[j]==0) ? 1.0 : 1.0 - idfnum / (idfden[i] * idfden[j]);
        sentenceTFIDFEuclidean[i][j] = sentenceTFIDFEuclidean[j][i] = Math.sqrt(euclideanSum);
      }
    }

    sentenceTFIDFCosineSim = tfidfsim;
    sentenceDissimilarities = idfdist;
    //sentenceDissimilarities = sentenceTFIDFEuclidean;

    return new DocMatrices(sentenceTFIDFCosineSim, sentenceDissimilarities, sentenceIDFVectors);
  }




  private void selectSentences(int summarySize, DocMatrices[] matrices, double lambda, List<String> sentences, LengthUnit lengthUnit, Collection<Flags> flags, double[] weights, double[][] sentenceVectorsForClustering, String idfVectorFileName, String docName)
  {
    HashSet<Integer> discarded = new HashSet<Integer>();
    HashSet<Integer> selected = new HashSet<Integer>();
    selectedList = new LinkedList<Integer>();
    DocMatrices avgMx = getAggregateMatrix(matrices, weights, selected.size(), flags);

    int K = getK(matrices[0].distances.length);

    LinkedList<int[]> clusterings = new LinkedList<int[]>();
    if(sentenceVectorsForClustering != null)
    {
      System.out.println("Clustering specified sentence vectors!");
      clusterings.add(getClusteringByVectors(sentenceVectorsForClustering, K, idfVectorFileName, docName));
    }
    else if(matrices[0].sentenceVectors != null)
    {
      System.out.println("Clustering sentence vectors!");
      clusterings.add(getClusteringByVectors(matrices[0].sentenceVectors, K, idfVectorFileName, docName));
    }
    else if(flags.contains(Flags.MOD_R1SUPER))
    {
      System.out.println("Clustering MOD_R1SUPER!");
      for(DocMatrices mp: matrices)
        clusterings.add(getClustering(mp, K));
    }
    else
    {
      System.out.println("Clustering average!");
      clusterings.add(getClustering(getAggregateMatrix(matrices, weights, selected.size(), flags), K));
    }

    while(summaryIsTooShort(selected, sentences, lengthUnit, summarySize))
    {
      if(weights != null)
      {
        avgMx = getAggregateMatrix(matrices, weights, selected.size(), flags);

        if(sentenceVectorsForClustering != null || matrices[0].sentenceVectors != null || flags.contains(Flags.MOD_R1SUPER))
        {
          //clusterings don't change in these cases. Not neccessary to recompute.
        }
        else
        {
          clusterings = new LinkedList<int[]>();
          System.out.println("Clustering average!");
          clusterings.add(getClustering(avgMx, K));
        }
      }
      //Integer secondBest = -1, thirdBest = -1;
      //double secondBestScore = 0.0, thirdBestScore = 0.0;

      double max = 0.0;
      Integer argmax = null;
      for (int i = 0; i < avgMx.similarities.length; ++i)
      {
        if(!selected.contains(i) && !discarded.contains(i))
        {
          selected.add(i);
          double curr = L1 (selected, avgMx.similarities, null, A) + lambda * R1Super(selected, avgMx.similarities, clusterings, K);
          /* as in Lin-Bilmes 2010: */
          if((lengthUnit == LengthUnit.CHARACTERS || lengthUnit == LengthUnit.WORDS) && (flags.contains(Flags.MOD_COST_WEIGHT)))
            curr /= Math.pow(sentences.get(i).length(),LINBILMES_CONSTANT_R);
          if (curr > max)
          {
            //thirdBest = secondBest; if(argmax != null) secondBest = argmax;
            //thirdBestScore = secondBestScore; secondBestScore = max;

            argmax = i;
            max = curr;
          }
          selected.remove(i);
        }
      }

      if (argmax != null)
      {
        selected.add(argmax); //internal: zero-based.
        selectedList.add(argmax+1); //outside visibility: one-based indexing.

        //System.out.println("Put "+(argmax+1)+" ("+max+") into summary. Second: "+(secondBest+1)+" ("+secondBestScore+"), third: "+(thirdBest+1)+" ("+thirdBestScore+").");
      }
      else
      {
        break;
      }

      if(flags.contains(Flags.MOD_STRICT_LENGTH))
      {
        if (argmax != null)
        {
          if(summaryIsTooLong(selected, sentences, lengthUnit, summarySize))
          {
            selected.remove(argmax);
            selectedList.removeLast();
            discarded.add(argmax);
          }
        }
      }
    }

    if(flags.contains(Flags.MOD_SINGLETON))
    {
      if(weights != null)
      {
        avgMx = getAggregateMatrix(matrices, weights, 0, flags);

        if(sentenceVectorsForClustering != null || matrices[0].sentenceVectors != null || flags.contains(Flags.MOD_R1SUPER))
        {
          //clusterings don't change in these cases. Not neccessary to recompute.
        }
        else
        {
          clusterings = new LinkedList<int[]>();
          System.out.println("Clustering average!");
          clusterings.add(getClustering(avgMx, K));
        }
      }
      double currentlyBestCScore = L1(selected, avgMx.similarities, null, A) + lambda * R1Super(selected, avgMx.similarities, clusterings, K);
      Integer currentlyBestSingleton = null;
      for(int i = 0; i < avgMx.similarities.length; i++)
      {
        HashSet<Integer> singleton = new HashSet<Integer>();
        singleton.add(i);
        if(!summaryIsTooLong(singleton, sentences, lengthUnit, summarySize))
        {
          double singletonSummaryScore = L1(singleton, avgMx.similarities, null, A) + lambda * R1Super(singleton, avgMx.similarities, clusterings, K);
          if(singletonSummaryScore > currentlyBestCScore)
          {
            currentlyBestCScore = singletonSummaryScore;
            currentlyBestSingleton = i;
          }
        }
      }
      if(currentlyBestSingleton != null)
      {
        selectedList = new LinkedList<Integer>();
        selectedList.add(currentlyBestSingleton+1);
      }
    }
  }

  private DocMatrices getAggregateMatrix(DocMatrices[] ms, double[] weights, int weightOffset, Collection<Flags> flags)
  {
    if(flags.contains(Flags.MOD_MATRIX_MULT))
      return getMultipliedAggregateSimilarities(ms);
    else if(flags.contains(Flags.MOD_MATRIX_L2))
      return getLpNormAggregateSimilarities(ms, 2.0);
    else if(flags.contains(Flags.MOD_MATRIX_L3))
      return getLpNormAggregateSimilarities(ms, 3.0);
    else if(flags.contains(Flags.MOD_MATRIX_L4))
      return getLpNormAggregateSimilarities(ms, 4.0);
    else if(flags.contains(Flags.MOD_MATRIX_L100))
      return getLpNormAggregateSimilarities(ms, 100.0);
    else if(flags.contains(Flags.MOD_MATRIX_MAX))
      return getMaxMatrix(ms);
    else if(flags.contains(Flags.MOD_MATRIX_MIN))
      return getMinMatrix(ms);
    else
      return getAverageMatrix(ms, weights, weightOffset);
  }
  public TreeSet<Integer> getSummaryAsTreeSet()
  {
    TreeSet<Integer> sorted = new TreeSet<Integer>();
    for(Integer i: selectedList)
      sorted.add(i);
    return sorted;
  }
  public List<Integer> getSummaryAsList()
  {
    return selectedList;
  }
  public String toString()
  {
    if(selectedList == null)
      return "Not initialized.";
    StringBuffer sb = new StringBuffer();
    for(Integer i: selectedList)
      sb.append(i+"\n");
    return sb.toString();
  }

  public String getSummaryAsText(List<String> sentences)
  {
    if(selectedList == null)
      return "Not initialized.";
    StringBuffer sb = new StringBuffer();
    for(Integer i: selectedList)
      sb.append(sentences.get(i-1)+"\n");
    return sb.toString();
  }

  /*
   * As described by lin-bilmes.
   * 
   */
  public static HashMap<String, Double> getIdfsDocClusterOrSentences(String clusterPath, List<String> sentences, String stopwordsFilename, String wordClusterFilename)
  {
    if(clusterPath != null)
      return getIdfsDocCluster(clusterPath, stopwordsFilename, wordClusterFilename);
    else if(sentences != null)
    {
      ArrayList<List<String>> l = new ArrayList<List<String>>(Arrays.asList(sentences));
      return getIdfsFromDocCollection(l, stopwordsFilename, wordClusterFilename);
    }
    return null;
  }
  public static HashMap<String, Double> getIdfsDocCluster(String clusterPath, String stopwordsFilename, String wordClusterFilename)
  {
    if(clusterPath == null)
      return null;
    System.out.println("Calculating IDFS.");

    List<List<String>> docCluster = new ArrayList<List<String>>();

    for(File docFile: new File(clusterPath).listFiles())
    {
      ArrayList<String> documentSentences = new ArrayList<String>();

      //There are gold standard summary directories.
      if(docFile.isDirectory())
        continue;
      String line;
      try
      {
        BufferedReader br = new BufferedReader(new FileReader(docFile));
        while ((line = br.readLine()) != null)
        {
          documentSentences.add(line);
        }
        br.close();
      }
      catch (IOException e)
      {
        System.err.println("Failed to open or read sentence files for IDF computations.");
        e.printStackTrace();
        System.exit(-1);
      }
      docCluster.add(documentSentences);
    }
    return getIdfsFromDocCollection(docCluster, stopwordsFilename, wordClusterFilename);

  }

  /*
   * Not lin-bilmes' way. But a reasonable way.
   */
  public static HashMap<String, Double> getIdfsDocCorpus(String corpusPath, String stopwordsFilename, String wordClusterFilename)
  {
    System.out.println("Calculating IDFS.");

    List<List<String>> corpus = new ArrayList<List<String>>();

    for(File docFolder: new File(corpusPath).listFiles())
    {
      ArrayList<String> docSentences = new ArrayList<String>();
      File sentenceFile = new File(docFolder.getAbsolutePath()+"/sentences");
      String line;
      try
      {
        BufferedReader br = new BufferedReader(new FileReader(sentenceFile));
        while ((line = br.readLine()) != null)
        {
          docSentences.add(line);
        }
        br.close();
      }
      catch (IOException e)
      {
        System.err.println("Failed to open or read sentence files for IDF computations.");
        e.printStackTrace();
        System.exit(-1);
      }
      corpus.add(docSentences);
    }
    return getIdfsFromDocCollection(corpus, stopwordsFilename, wordClusterFilename);
  }

  public static HashMap<String, Double> getIdfsFromDocCollection(List<List<String>> documentCluster, String stopwordsFilename, String wordClusterFilename)
  {
    HashMap<String, Integer> documentCountsForTerm = new HashMap<String, Integer>();

    HashSet<String> stopwords = getStopwords(stopwordsFilename);
    HashMap<String, String> wordClusterId = getWordClusters(wordClusterFilename);

    // Special case: if all sentences are from one file, idfs from one document make no sense.
    // (similarities would be zero).
    // Will then consider each sentence as one document.
    if(documentCluster.size() <= 1)
    {
      List<List<String>> newCluster = new ArrayList<List<String>>();
      List<String> document = documentCluster.get(0);
      for(String s: document)
      {
        List<String> l = new ArrayList<String>();
        l.add(s);
        newCluster.add(l);
      }
      documentCluster = newCluster;
    }

    for (List<String> document: documentCluster)
    {
      HashSet<String> curDoc = new HashSet<String>();
      for (String sentence: document)
      {
        if (sentence.length() > 0)
        {
          String[] words = sentence.split(REGEX_SPACE);
          String prev = null;
          for(String w: words)
          {
            if(w.replace("_", "").isEmpty())
              continue;

            w = w.toLowerCase();

            PorterStemmer stemmer = new PorterStemmer();
            stemmer.addString(w);
            stemmer.stem();
            String stemmed = stemmer.toString();

            // We always include bigrams when computing IDFS. These can be ignored later,
            // but Lin&Bilmes used them.
            if(prev != null)
            {
              curDoc.add(prev+" "+stemmed);

              // olof: replace words with their cluster id:
              if (wordClusterId.containsKey(prev) || wordClusterId.containsKey(stemmed))
              {
                //System.out.println("replacing word "+stemmed+" with its cluster id: "+wordClusterId[stemmed]);
                String clusterReplacedBigram = "";
                if(wordClusterId.containsKey(prev))
                  clusterReplacedBigram = wordClusterId.get(prev);
                else
                  clusterReplacedBigram = prev;

                if(wordClusterId.containsKey(stemmed))
                  clusterReplacedBigram += " "+wordClusterId.get(stemmed);
                else
                  clusterReplacedBigram += " "+stemmed;

                curDoc.add(clusterReplacedBigram);
              }
            }
            //end, bigrams

            if(!stopwords.contains(w))
            {
              curDoc.add(stemmed);

              //also add cluster-replacement!
              if(wordClusterId.containsKey(stemmed))
                curDoc.add(wordClusterId.get(stemmed));

              prev = stemmed;
            }
            else
              prev = w;
          }
        }
      }

      for(String term: curDoc)
      {
        if(documentCountsForTerm.containsKey(term))
          documentCountsForTerm.put(term, documentCountsForTerm.get(term)+1);
        else
          documentCountsForTerm.put(term, 1);
      }
    }
    HashMap<String, Double> idfs = new HashMap<String, Double>();
    for(String term: documentCountsForTerm.keySet())
    {
      double idf = Math.log10(((double)documentCluster.size())/documentCountsForTerm.get(term).doubleValue());
      idfs.put(term, idf);
    }
    System.out.println("Done calculating IDFS.");

    return idfs;
  }

  public static void recursivelyCreateDirectory(File f) throws IOException
  {
    if(f != null && !f.exists())
    {
      recursivelyCreateDirectory(f.getParentFile());
      f.mkdir();
    }
  }
  protected static double getDefaultLambda()
  {
    return DEFAULT_LAMBDA;
  }
}
