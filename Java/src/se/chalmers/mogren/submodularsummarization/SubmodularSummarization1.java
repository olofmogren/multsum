package se.chalmers.mogren.submodularsummarization;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeSet;

import se.chalmers.mogren.submodularsummarization.util.PorterStemmer;

public class SubmodularSummarization1 extends SubmodularSummarization
{
  private static final String USAGE_STRING = "Usage: <command> --matrix-file <filename1> [--matrix-file <filename2> [--matrix-file <filename3> [...]]] \n" +
      "--document-file file \n" +
      "--stopwords-file filename \n" +
      "--word-clusters-file filename \n" +
      "[--nb-sentences summarysize] \n" +
      "[--nb-words summarysize] \n" +
      "[--nb-characters summarysize] \n" +
      "[--lambda lambda] \n" +
      "(Either at least one matrix file or a document file must be present.)\n";

  private static final String REGEX_SPACE         = "\\W+";
  private static final String REGEX_SPACE_COMMA   = "\\s+|,";

  private static final String REGEX_NONWORD       = "[^\\p{L}\\p{Nd}]+";

  private static final double ALPHA                = 0.75f;
  private static final double DEFAULT_LAMBDA       = 6.0f;

  private static final double LINBILMES_CONSTANT_R = 0.3f;
  LinkedList<Integer> selectedList;

  public static void main(String[] args)
  {
    SubmodularSummarization1 summary = null;
    double lambda = DEFAULT_LAMBDA;
    int summarySize = 2;
    LengthUnit lengthUnit = LengthUnit.SENTENCES;
    String stopwordsFilename = null;
    String documentFileName = null;
    String wordClusterFileName = null;

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
        lengthUnit = LengthUnit.SENTENCES;
      }
      else if(args[i].equals("--nb-words"))
      {
        summarySize = Integer.parseInt(args[++i]);
        lengthUnit = LengthUnit.WORDS;
      }
      else if(args[i].equals("--nb-characters"))
      {
        summarySize = Integer.parseInt(args[++i]);
        lengthUnit = LengthUnit.CHARACTERS;
      }
      else if(args[i].equals("--lambda"))
      {
        lambda = (double)Integer.parseInt(args[++i]);
      }
      else
      {
        System.out.println(USAGE_STRING);
        System.exit(0);
      }
    }

    ArrayList<String> sentences = null;
    if(documentFileName != null)
    {
      sentences =  getSentencesList(documentFileName);
    }

    if(matrixFileNames.size() > 0)
    {
      if(sentences == null)
        System.err.println("WARNING! sentences list is null!");

      summary = new SubmodularSummarization1(matrixFileNames.toArray(new String[matrixFileNames.size()]), sentences, lambda, summarySize, lengthUnit, new HashSet<Flags>());
    }
    else
    {
      summary = new SubmodularSummarization1(stopwordsFilename, documentFileName, wordClusterFileName, lambda, summarySize, lengthUnit, new HashSet<Flags>());
    }

    System.out.println(summary);
  }
  public SubmodularSummarization1(String stopwordsFilename, String documentFileName, String wordClusterFileName, Double lambda, int summarySize, LengthUnit lengthUnit, Collection<Flags> flags)
  {
    if(lambda == null)
      lambda = DEFAULT_LAMBDA;
    DocMatrices sentenceMeasures = getDefaultSentenceSimilarities(stopwordsFilename, documentFileName, wordClusterFileName, (flags.contains(Flags.MOD_BIGRAMS)));
    
    if(flags.contains(Flags.OPTION_SAVE_MATRICES))
    {
      try
      {
        saveLinTFIDFMatrix(sentenceMeasures.similarities, new File(documentFileName).getParentFile().getName());
      }
      catch(IOException e)
      {
        e.printStackTrace();
      }
    }
    
    int K = getK(sentenceMeasures.distances.length);
    LinkedList<int[]> clusterings = new LinkedList<int[]>();
    clusterings.add(getClustering(sentenceMeasures, K));
    ArrayList<String> sentences = getSentencesList(documentFileName);

    selectSentences(summarySize, sentenceMeasures.similarities, lambda, clusterings, K, sentences, lengthUnit, flags);
  }

  public SubmodularSummarization1(String[] matrixFileNames, ArrayList<String> sentences, Double lambda, int summarySize, LengthUnit lengthUnit, Collection<Flags> flags)
  {
    if(lambda == null)
      lambda = DEFAULT_LAMBDA;
    ArrayList<double[][]> sims = new ArrayList<double[][]>();
    for(String f: matrixFileNames)
    {
      sims.add(getSimilaritiesFromFile(f));
    }

    DocMatrices[] matrices = new DocMatrices[sims.size()];
    for(int i = 0; i < sims.size(); i++)
    {
      matrices[i] = new DocMatrices(sims.get(i), getDissimilarities(sims.get(i)));
    }
    
    DocMatrices sentenceSimilarities = null;
    if(matrixFileNames.length > 1)
      sentenceSimilarities = getAverageMatrix(matrices, null, 0);
    else
      sentenceSimilarities = new DocMatrices(sims.get(0), getDissimilarities(sims.get(0)));
      
    int K = getK(sentenceSimilarities.similarities.length);
    //int[] clustering = getClustering(sentenceDissimilarities, K);


    LinkedList<int[]> clusterings = new LinkedList<int[]>();
    if(flags.contains(Flags.MOD_R1SUPER))
    {
      for(DocMatrices matrix: matrices)
      {
        clusterings.add(getClustering(matrix, K));
      }
    }
    else
      clusterings.add(getClustering(sentenceSimilarities, K));

    selectSentences(summarySize, sentenceSimilarities.similarities, lambda, clusterings, K, sentences, lengthUnit, flags);
  }

  private static ArrayList<String> getSentencesList(String documentFileName)
  {
    ArrayList<String> sentencesList = new ArrayList<String>();
    try
    {
      BufferedReader reader = new BufferedReader(new FileReader(documentFileName));
      String line = reader.readLine();
      while (line != null)
      {
        sentencesList.add(line);
        line = reader.readLine();
      }
      reader.close();
    }
    catch(IOException e)
    {
      e.printStackTrace();
      System.exit(-1);
    }
    return sentencesList;
  }

  private DocMatrices getDefaultSentenceSimilarities(String stopwordsFilename, String documentFileName, String wordClusterFileName, boolean includeBigrams)
  {
    //Some initializations:
    double[][] sentenceSimilarities = null;
    double[][] sentenceDissimilarities = null;

    //System.out.println ("Using internal TFIDF-based sentence similaritites.");
    HashSet<String> stopwords = new HashSet<String> ();
    try
    {
      BufferedReader reader = new BufferedReader(new FileReader(stopwordsFilename));
      String line = reader.readLine();
      while (line != null)
      {
        stopwords.add(line);
        line = reader.readLine ();
      }
      reader.close();
    }
    catch (Exception e)
    {
      e.printStackTrace();
      System.exit(1);
    }

    HashMap<String, String> wordClusterId = getWordClusters(wordClusterFileName);

    try
    {
      HashMap<String, double[]> document = new HashMap<String, double[]> ();
      ArrayList<HashMap<String, Integer>> sentencesList = new ArrayList<HashMap<String, Integer>> ();
      BufferedReader reader = new BufferedReader(new FileReader(documentFileName));
      String line = reader.readLine();
      while (line != null)
      {
        if (line.length() > 0)
        {
          HashMap<String, Integer> currsentence = new HashMap<String, Integer> ();
          String[] words = line.split(REGEX_SPACE);
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

            // olof: replace words with their cluster id:
            if (wordClusterId.containsKey(stemmed))
            {
              //System.out.println("replacing word "+stemmed+" with its cluster id: "+wordClusterId[stemmed]);
              stemmed = wordClusterId.get(stemmed);
            }

            if(includeBigrams && prev != null)
            {
              String bigram = prev+" "+stemmed;

              if (document.containsKey(bigram))
              {
                double[] f = document.get(bigram);
                f[0]++;
                document.put(bigram, f);

              }
              else
              {
                double[] defaultValues = new double[2]; defaultValues[0] = 1.0f; defaultValues[1] = 0.0f;
                document.put(bigram, defaultValues);
              }
              if (currsentence.containsKey(bigram))
              {
                Integer i = currsentence.get(bigram);
                currsentence.put(bigram, ++i);
              }
              else
              {
                currsentence.put(bigram, 1);
                double[] f = document.get(bigram);
                f[1]++;
                document.put(bigram, f);
              }
            }
            //end, bigrams

            if(stopwords.contains(w) == false)
            {
              if (document.containsKey(stemmed))
              {
                double[] f = document.get(stemmed);
                f[0]++;
                document.put(stemmed, f);

              }
              else
              {
                double[] defaultValues = new double[2]; defaultValues[0] = 1.0f; defaultValues[1] = 0.0f;
                document.put(stemmed, defaultValues);
              }
              if (currsentence.containsKey(stemmed))
              {
                Integer i = currsentence.get(stemmed);
                currsentence.put(stemmed, ++i);
              }
              else
              {
                currsentence.put(stemmed, 1);
                double[] f = document.get(stemmed);
                f[1]++;
                document.put(stemmed, f);
              }
              prev = stemmed;
            }
            else
              prev = w;
          }
          sentencesList.add(currsentence);
        }
        line = reader.readLine ();
      }
      reader.close();

      //HashMap<String, Integer>[] sentences = new  HashMap<String, Integer>[sentencesList.size()];
      //for(int i = 0; i < sentencesList.size(); i++)
      //  sentencesList.get(i) = sentencesList.get(i);

      double[] tfidfden = new double[sentencesList.size()];
      double[] idfden = new double[sentencesList.size()];
      for (int i = 0; i != sentencesList.size(); ++i) {
        tfidfden [i] = idfden [i] = 0.0f;
        for(String term: sentencesList.get(i).keySet())
        {
          Integer value = sentencesList.get(i).get(term);
          double idf = document.get(term)[1];
          tfidfden [i] += value * value * idf * idf;
          idfden [i] += idf * idf;
        }
        tfidfden [i] = (double)Math.sqrt(tfidfden [i]);
        idfden [i] = (double)Math.sqrt(idfden [i]);
      }

      double[][] tfidfsim = new double[sentencesList.size()][sentencesList.size()];
      double[][] idfdist = new double[sentencesList.size()][sentencesList.size()];
      for (int i = 0; i != sentencesList.size(); ++i)
      {
        tfidfsim [i][i] = 1.0f;
        idfdist [i][i] = 0.0f;
        for (int j = i + 1; j != sentencesList.size(); ++j) {
          double tfidfnum = 0.0f, idfnum = 0.0f;
          for(String term: sentencesList.get(i).keySet())
          {
            if (sentencesList.get(j).containsKey(term))
            {
              Integer value = sentencesList.get(i).get(term);
              double idf = document.get(term)[1];
              tfidfnum += value * idf * sentencesList.get(j).get(term) * idf;
              idfnum += idf * idf;
            }
          }
          tfidfsim[i][j] = tfidfsim[j][i] = (tfidfden[i]==0 || tfidfden[j]==0) ? 0.0f : tfidfnum / (tfidfden[i] * tfidfden[j]);
          idfdist[i][j] = idfdist[j][i] = (idfden[i]==0 || idfden[j]==0) ? 1.0f : 1.0f - idfnum / (idfden[i] * idfden[j]);

        }
      }

      sentenceSimilarities = tfidfsim;
      sentenceDissimilarities = idfdist;

    }
    catch (Exception e)
    {
      e.printStackTrace();
      System.exit(2);
    }

    try
    {
      BufferedWriter bw = new BufferedWriter(new FileWriter(new File(System.getProperty("user.home")+"/tmp/submod1_matrix_sim.mat")));
      for(double[] row: sentenceSimilarities)
      {
        boolean first = true;
        for(double cell: row)
        {
          if(first)
            first = false;
          else
            bw.write(", ");
          bw.write(cell+"");
        }
        bw.write("\n");
      }
      bw.close();
    }
    catch(IOException e)
    {
      e.printStackTrace();
    }

    return new DocMatrices(sentenceSimilarities, sentenceDissimilarities);
  }

  private void selectSentences(int summarySize, double[][] sentenceSimilarities, double lambda, Collection<int[]> clusterings, int K, ArrayList<String> sentences, LengthUnit lengthUnit, Collection<Flags> flags)
  {
    if(clusterings.size() > 1)
    {
      int mismatchCount = 0;
      for(int[] clustering1: clusterings)
      {
        for(int[] clustering2: clusterings)
        {
          if(clustering1 == clustering2)
            continue;
          for(int i1 = 0; i1 < clustering1.length; i1++)
          {
            if(clustering1[i1] != clustering2[i1])
              mismatchCount++;
          }
        }        
      }
      System.out.println("Clusterings differ at "+mismatchCount+" places. Number of clusterings: "+clusterings.size()+".");
    }

    HashSet<Integer> discarded = new HashSet<Integer>();
    HashSet<Integer> selected = new HashSet<Integer>();
    selectedList = new LinkedList<Integer>();
    while(summaryIsTooShort(selected, sentences, lengthUnit, summarySize))
    {
      Integer secondBest = -1, thirdBest = -1;
      double secondBestScore = 0.0f, thirdBestScore = 0.0f;

      double max = 0.0f;
      Integer argmax = null;
      for (int i = 0; i != sentenceSimilarities.length; ++i)
      {
        if(!selected.contains(i) && !discarded.contains(i))
        {
          selected.add(i);
          double curr = L1(selected, sentenceSimilarities, ALPHA, null) + lambda * R1Super(selected, sentenceSimilarities, clusterings, K);
          /* as in Lin-Bilmes 2010: */
          if(lengthUnit == LengthUnit.CHARACTERS && (flags.contains(Flags.MOD_COST_WEIGHT)))
            curr /= Math.pow(sentences.get(i).length(),LINBILMES_CONSTANT_R);
          if (curr > max)
          {
            thirdBest = secondBest; if(argmax != null) secondBest = argmax;
            thirdBestScore = secondBestScore; secondBestScore = max;

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
      double currentlyBestCScore = L1(selected, sentenceSimilarities, ALPHA, null) + lambda * R1Super(selected, sentenceSimilarities, clusterings, K);
      Integer currentlyBestSingleton = null;
      for(int i = 0; i < sentenceSimilarities.length; i++)
      {
        HashSet<Integer> singleton = new HashSet<Integer>();
        singleton.add(i);
        if(!summaryIsTooLong(singleton, sentences, lengthUnit, summarySize))
        {
          double singletonSummaryScore = L1(singleton, sentenceSimilarities, ALPHA, null) + lambda * R1Super(singleton, sentenceSimilarities, clusterings, K);
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

  protected static double getDefaultLambda()
  {
    return DEFAULT_LAMBDA;
  }
}
