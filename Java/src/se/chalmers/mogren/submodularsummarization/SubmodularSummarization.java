/**
 * 
 */
package se.chalmers.mogren.submodularsummarization;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.management.ManagementFactory;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import se.chalmers.mogren.submodularsummarization.util.KMeans;
import se.chalmers.mogren.submodularsummarization.util.PorterStemmer;

/**
 * 
 * $author Olof Mogren, mogren@chalmers.se
 * 
 * Base class for submodular summarization.
 * Providing some base functions.
 *
 * Released under the GNU GPL.
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

public abstract class SubmodularSummarization
{
  public enum Flags {MOD_COST_WEIGHT,
    MOD_SINGLETON,
    MOD_BIGRAMS,
    MOD_STRICT_LENGTH,
    OPTION_SAVE_MATRICES,
    MOD_WEIGHTED_AVERAGE,
    MOD_R1SUPER,
    MOD_MATRIX_MAX,
    MOD_MATRIX_MIN,
    MOD_MATRIX_MULT,
    MOD_MATRIX_L2,
    MOD_MATRIX_L3,
    MOD_MATRIX_L4,
    MOD_MATRIX_L100
  };

  public enum LengthUnit {SENTENCES, WORDS, CHARACTERS};

  public static final String REGEX_SPACE         = "\\W+";
  public static final String REGEX_SPACE_COMMA   = "\\s+|,";
  public static final String REGEX_NONWORD       = "[^\\p{L}\\p{Nd}]+";

  public static final String CLUTO_SCLUSTER_EXECUTABLE = "/home/mogren/Syncthing/Code/others_code/cluto/scluster";
  public static final String CLUTO_VCLUSTER_EXECUTABLE = "/home/mogren/Syncthing/Code/others_code/cluto/vcluster";

  protected double L1(HashSet<Integer> S, double[][] w, Double alpha, Double a)
  {
    if(alpha == null)
      alpha = a/(double)w.length;
    double res = 0.0;
    for (int i = 0; i < w.length; ++i)
    {
      double sum = 0.0, sumV = 0.0;
      for(Integer j: S)
        sum += w[i][j];
      for (int k = 0; k < w.length; ++k)
        sumV += w[i][k];
      sumV *= alpha;
      res += Math.min(sum, sumV);
    }
    return res;
  }

  /*  protected double R1(HashSet<Integer> S, double[][] w, int[] clustering, int K)
  {
    int N = w.length;
    double res = 0.0;
    for (int k = 0; k != K; ++k)
    {
      double sum = 0.0;
      for(Integer j: S)
        if (clustering [j] == k)
        {
          // sumV is called r_j in paper.
          double sumV = 0.0;
          for (int i = 0; i != N; ++i)
            sumV += w [i][j];
          sum += sumV / N;
        }
      res += (double)Math.sqrt(sum);
    }
    return res;
  }*/

  /*
   * An attempt to create a better R1 function. Will check for diversity in a summary S,
   * not only according to one clustering, but to a collection of different clusterings.
   * The summands (added into sum, which square root is added to res) are also divided
   * by the number of clusterings. If all clusterings agree, the value should be the
   * same as in the original R1. Each clustering will provide a smaller amount.
   * In this way, we should be able to keep weights, alpha, lambda as is.
   * (Specifically, if clusterings.size() == 1, R1Super == R1).
   * 
   */
  protected double R1Super(HashSet<Integer> S, double[][] w, Collection<int[]> clusterings, int K)
  {
    double res = 0.0;
    if(S.size() == 1)//Singleton reward:
    {
      double sum = 0.0;
      for(Integer j: S)
      {
        for(int i = 0; i < w.length; i++)
          sum += w[j][i];
      }
      res = sum/(double)w.length;
    }
    else
    {
      for (int k = 0; k != K; ++k)
      {
        double sum = 0.0;
        for(Integer j: S)
        {
          for(int[] clustering: clusterings)
          {
            if (clustering[j] == k)
            {
              // sumV is called r_j in paper.
              double sumV = 0.0;
              for (int i = 0; i < w.length; ++i)
                sumV += w [i][j];
              sum += sumV / (w.length*clusterings.size());
            }
          }
        }
        res += (double)Math.sqrt(sum);
      }
    }
    return res;
  }

  protected static double[][] getSimilaritiesFromFile(String similarityMatrixFileName)
  {
    double[][] sentenceSimilarities = null;

    try
    {
      BufferedReader simReader = new BufferedReader(new FileReader(similarityMatrixFileName));
      String simLine = simReader.readLine();
      int lineNumber = 0;
      while (simLine != null)
      {
        simLine = simLine.replace(";", "");
        String[] words = simLine.split(REGEX_SPACE_COMMA);
        if(sentenceSimilarities == null)
        {
          int length = 0;
          for(String word: words)
          {
            if(!word.replace("_", "").isEmpty())
              length++;
          }
          sentenceSimilarities = new double[length][length];
        }

        int wordNumber = 0;
        for(String word: words)
        {
          //System.out.println(word+" "+word.length()+" "+wordNumber);
          if(!word.replace("_", "").isEmpty())
          {
            double value = Double.parseDouble(word);
            sentenceSimilarities [lineNumber][wordNumber] = value;
            wordNumber++;
          }
        }
        if (wordNumber != sentenceSimilarities.length)
        {
          System.err.println("Mismatch: found " + wordNumber + " similaritities (cols), but there are " + sentenceSimilarities.length + " sentences.");
          System.exit(1);
        }
        //System.in.read();
        simLine = simReader.readLine();
        lineNumber++;
      }

      if (lineNumber != sentenceSimilarities.length)
      {
        System.err.println("Mismatch: found " + lineNumber + " similaritities (lines), but there are " + sentenceSimilarities.length + " sentences.");
        System.exit(1);
      }
      simReader.close();
    }
    catch (Exception e)
    {
      e.printStackTrace();
      System.exit(1);
    }
    return sentenceSimilarities;
  }

  protected static double[] getWeightsFromFile(String weightsFileName)
  {
    double[] weights = null;

    try
    {
      BufferedReader fReader = new BufferedReader(new FileReader(weightsFileName));
      String simLine = fReader.readLine();
      int lineNumber = 0;
      while (simLine != null)
      {
        if(lineNumber > 0)
        {
          System.err.println("Only considering weights files one line long.");
          break;
        }
        String[] words = simLine.split(REGEX_SPACE_COMMA);
        if(weights == null)
        {
          weights = new double[words.length];
        }

        int wordNumber = 0;
        for(String word: words)
        {
          //System.out.println(word+" "+word.length()+" "+wordNumber);
          if(!word.isEmpty())
          {
            double value = Double.parseDouble(word);
            weights[wordNumber] = value;
            wordNumber++;
          }
        }
        if (wordNumber != weights.length)
        {
          double[] newWeights = new double[wordNumber];
          for(int i = 0; i < wordNumber; i++)
          {
            newWeights[i] = weights[i];
          }
          weights = newWeights;
        }
        //System.in.read();
        simLine = fReader.readLine();
        lineNumber++;
      }
      fReader.close();
    }
    catch (Exception e)
    {
      e.printStackTrace();
      System.exit(1);
    }
    return weights;
  }

  protected double[][] getDissimilarities(double[][] similarities)
  {
    double[][] dissimilarities = new double[similarities.length][similarities.length];
    for (int i=0; i < similarities.length; i++)
    {
      for (int j=0; j<similarities.length; j++)
      {
        dissimilarities [i][j] = 1.0 - similarities[i][j];
      }
    }
    return dissimilarities;
  }

  protected int getK(int N)
  {
    return (int)(0.2f * (double)N + 0.5f);
  }

  protected int[] getClustering(DocMatrices mp, int K)
  {
    return getClusteringCluto(mp.similarities, K);
  }
  /*  protected int[] getClusteringKMedians(double[][] distances, int K)
  {
    KMeans kmeans = new KMeans(distances.length, K);
    return kmeans.Cluster(distances);
    //kmeans.Show ();
  }*/

  protected int[] getClusteringCluto(double[][] adjacencies, int K)
  {
    int[] clustering = null;
    String tmpfilename = "/tmp/submod_matrix_to_cluster"+
        ManagementFactory.getRuntimeMXBean().getName()+
        Thread.currentThread().getId()+".mat"; 
    String outputfilename = tmpfilename+".clustering."+K;
    try {
      BufferedWriter bw = new BufferedWriter(new FileWriter(tmpfilename));
      bw.write(adjacencies.length+"\n");
      for(int i=0; i<adjacencies.length; i++)
      {
        for(int j=0; j<adjacencies.length; j++)
        {
          bw.write(adjacencies[i][j]+"");
          if(j < adjacencies[0].length-1)
            bw.write(" ");
        }
        bw.write("\n");
      }
      bw.close();

      LinkedList<String> commandList = new LinkedList<String>();
      commandList.add(CLUTO_SCLUSTER_EXECUTABLE);
      //commandList.add("-crfun=i1");
      //commandList.add("-clmethod=direct");
      commandList.add(tmpfilename);
      commandList.add(K+"");

      ProcessBuilder builder1 = new ProcessBuilder(commandList);

      //builder.directory(new File(ROUGE_EVAL_HOME_LOCATION));

      System.out.print("command from builder: ");
      for(String sc: builder1.command())
        System.out.print(sc+" ");
      System.out.println();

      builder1.redirectErrorStream(true);
      Process p1 = builder1.start();


      BufferedReader bri1 = new BufferedReader(new InputStreamReader(p1.getInputStream()));

      StringBuffer output = new StringBuffer();
      String line;
      while((line = bri1.readLine()) != null)
      {
        output.append(line+"\n");
      }

      boolean finished = false;
      while(!finished)
      {
        try
        {
          p1.waitFor();
          finished = true;
        }
        catch (InterruptedException e)
        {
          e.printStackTrace();
          System.err.println("Will continue waiting for process to terminate.");
        }
      }

      int ev = p1.exitValue();
      if(ev != 0)
      {
        System.err.println("CLUTO exited with error!\n"+output);
        System.exit(-1);
      }

      BufferedReader br = new BufferedReader(new FileReader(outputfilename));
      clustering = new int[adjacencies.length];

      int index = 0;
      while((line = br.readLine()) != null)
      {
        //System.out.println("Clustering from CLUTO: "+index+": "+line);

        clustering[index++] = Integer.parseInt(line);
      }
      br.close();

      new File(tmpfilename).delete();
      new File(outputfilename).delete();
    }
    catch (IOException e)
    {
      e.printStackTrace();
      System.exit(-1);
    }
    return clustering;
  }

  protected int[] getClusteringByVectors(double[][] sentenceVectors, int K, String vectorFileName, String docName)
  {
    int[] clustering = null;
    if(vectorFileName != null)
      vectorFileName = "/tmp/submod_idfvecs."+
        //ManagementFactory.getRuntimeMXBean().getName()+
        //Thread.currentThread().getId()+
        docName+
        ".mat";
    String outputfilename = vectorFileName+".clustering."+K;

    try
    {
      if(!new File(outputfilename).exists())
      {
        BufferedWriter bw = new BufferedWriter(new FileWriter(vectorFileName));
        //First rows, then columns:
        bw.write(sentenceVectors.length+" "+sentenceVectors[0].length+"\n");
        for(int i=0; i<sentenceVectors.length; i++)
        {
          for(int j=0; j<sentenceVectors[0].length; j++)
          {
            bw.write(sentenceVectors[i][j]+"");
            if(j < sentenceVectors[0].length-1)
              bw.write(" ");
          }
          bw.write("\n");
        }
        bw.close();

        LinkedList<String> commandList = new LinkedList<String>();
        commandList.add(CLUTO_VCLUSTER_EXECUTABLE);
        //commandList.add("-crfun=i1");
        //commandList.add("-clmethod=direct");

        //commandList.add("-clmethod=graph");
        //commandList.add("-sim=dist");

        commandList.add(vectorFileName);
        commandList.add(K+"");

        ProcessBuilder builder1 = new ProcessBuilder(commandList);

        //builder.directory(new File(ROUGE_EVAL_HOME_LOCATION));

        System.out.print("command from builder: ");
        for(String sc: builder1.command())
          System.out.print(sc+" ");
        System.out.println();

        builder1.redirectErrorStream(true);
        Process p1 = builder1.start();


        BufferedReader bri1 = new BufferedReader(new InputStreamReader(p1.getInputStream()));

        StringBuffer output = new StringBuffer();
        String line;
        while((line = bri1.readLine()) != null)
        {
          output.append(line+"\n");
        }

        boolean finished = false;
        while(!finished)
        {
          try
          {
            p1.waitFor();
            finished = true;
          }
          catch (InterruptedException e)
          {
            e.printStackTrace();
            System.err.println("Will continue waiting for process to terminate.");
          }
        }

        int ev = p1.exitValue();
        if(ev != 0)
        {
          System.err.println("CLUTO exited with error!\n"+output);
          System.exit(-1);
        }
        new File(vectorFileName).delete();
      }
      else
      {
        System.out.println("Using precomputed idf-clusters for "+docName);
      }

      BufferedReader br = new BufferedReader(new FileReader(outputfilename));
      clustering = new int[sentenceVectors.length];

      String line = null;
      int index = 0;
      while((line = br.readLine()) != null)
      {
        //System.out.println("Clustering from CLUTO: "+index+": "+line);

        clustering[index++] = Integer.parseInt(line);
      }
      br.close();
      //new File(outputfilename).delete();
    }
    catch (IOException e)
    {
      e.printStackTrace();
      System.exit(-1);
    }

    return clustering;
  }


  /*
   * 
   * weights contains a multiple of sets of weights, oach one set corresponding to the set of matrices ms. So the size of weights must be a multiple of the size of ms.
   * weightOffset is how far into weights we'll look. 
   * ms the matrices.
   * 
   */
  protected DocMatrices getAverageMatrix(DocMatrices[] ms, double[] weights, int weightOffset)
  {
    if(ms.length == 1)
      return ms[0];
    double[][] avgSim = null;
    double[][] avgDist = null;
    int index = 0;
    for(DocMatrices m: ms)
    {
      //System.out.println("("+m.length+", "+m[0].length+")");

      if(weights == null || weightOffset+index >= weights.length || weights[weightOffset+index] > 0.0) //a little speed-up. ignore all matrices with weight zero.
      {
        double[][] mSim = m.similarities;
        double[][] mDist = m.distances;
        if(avgSim == null)
        {
          avgSim = new double[mSim.length][mSim[0].length];
          avgDist = new double[mDist.length][mDist[0].length];

          for(int i=0; i < mSim.length; i++)
          {
            for(int j=0; j < mSim[i].length; j++)
            {
                avgSim[i][j] += 0.0;
                avgDist[i][j] += 0.0;
            }
          }
        }
        for(int i=0; i < mSim.length; i++)
        {
          for(int j=0; j < mSim[i].length; j++)
          {
            if(weights != null && weightOffset+index < weights.length)
            {
              avgSim[i][j] += mSim[i][j] * weights[weightOffset+index];
              avgDist[i][j] += mDist[i][j] * weights[weightOffset+index];
            }
            else
            {
              avgSim[i][j] += mSim[i][j] / (double)ms.length;
              avgDist[i][j] += mDist[i][j] / (double)ms.length;
            }
          }
        }
        if(weights != null && weightOffset+index < weights.length && weights[weightOffset+index] == 1.0) //a little speed-up. if this matrix has weight 1.0, no other matrix can have weight > 0.0. ignore all matrices with weight zero.
          break;
      }
      index++;
    }
    return new DocMatrices(avgSim, avgDist);
  }

  protected DocMatrices getLpNormAggregateSimilarities(DocMatrices[] ms, double p)
  {
    if(ms.length == 1)
      return ms[0];

    double[][] aggregateSim = null;
    double[][] aggregateDist = null;

    for(DocMatrices m: ms)
    {
      //System.out.println("("+m.length+", "+m[0].length+")");

      double[][] mSim = m.similarities;
      double[][] mDist = m.distances;
      if(aggregateSim == null)
      {
        aggregateSim = new double[mSim.length][mSim[0].length];
        aggregateDist = new double[mDist.length][mDist[0].length];
        for(int i=0; i < mSim.length; i++)
        {
          for(int j=0; j < mSim[i].length; j++)
          {
            aggregateSim[i][j] = 0.0;
            aggregateDist[i][j] = 0.0;
          }
        }
      }
      for(int i=0; i < mSim.length; i++)
      {
        for(int j=0; j < mSim[i].length; j++)
        {
          aggregateSim[i][j] += Math.pow(mSim[i][j], p);
          aggregateDist[i][j] += Math.pow(mDist[i][j], p);
        }
      }
    }

    //Get the p-root:
    for(int i=0; i < aggregateSim.length; i++)
    {
      for(int j=0; j < aggregateSim[i].length; j++)
      {
        aggregateSim[i][j] = Math.pow(aggregateSim[i][j], 1.0/p);
        aggregateDist[i][j] = Math.pow(aggregateDist[i][j], 1.0/p);
      }
    }

    return new DocMatrices(normalize(aggregateSim), normalize(aggregateDist));
  }

  private double[][] normalize(double[][] m)
  {
    double max = 0.0;
    double[][] mr = new double[m.length][m[0].length];
    //Get the  max:
    for(int i=0; i < m.length; i++)
    {
      for(int j=0; j < m[i].length; j++)
      {
        if(m[i][j] > max)
          max = m[i][j];
      }
    }

    //Normalize:
    for(int i=0; i < m.length; i++)
    {
      for(int j=0; j < m[i].length; j++)
      {
        mr[i][j] = m[i][j]/max;
      }
    }
    return mr;
  }
  
  protected DocMatrices getMultipliedAggregateSimilarities(DocMatrices[] ms)
  {
    if(ms.length == 1)
      return ms[0];

    double[][] aggregateSim = null;
    double[][] aggregateDist = null;

    for(DocMatrices m: ms)
    {
      //System.out.println("("+m.length+", "+m[0].length+")");

      double[][] mSim = m.similarities;
      double[][] mDist = m.distances;
      if(aggregateSim == null)
      {
        aggregateSim = new double[mSim.length][mSim[0].length];
        aggregateDist = new double[mDist.length][mDist[0].length];
        for(int i=0; i < mSim.length; i++)
        {
          for(int j=0; j < mSim[i].length; j++)
          {
            aggregateSim[i][j] = 1.0;
            aggregateDist[i][j] = 1.0;
          }
        }
      }
      for(int i=0; i < mSim.length; i++)
      {
        for(int j=0; j < mSim[i].length; j++)
        {
          aggregateSim[i][j] *= mSim[i][j];
          aggregateDist[i][j] *= mDist[i][j];
        }
      }
    }

    return new DocMatrices(aggregateSim, aggregateDist);
  }

  protected DocMatrices getMinMatrix(DocMatrices[] ms)
  {
    return getMinMaxMatrix(ms, false);
  }

  protected DocMatrices getMaxMatrix(DocMatrices[] ms)
  {
    return getMinMaxMatrix(ms, true);
  }

  private DocMatrices getMinMaxMatrix(DocMatrices[] ms, boolean max)
  {
    if(ms.length == 1)
      return ms[0];
    double[][] aggregateSim = null;
    double[][] aggregateDist = null;
    boolean firstMatrix = true;
    for(DocMatrices m: ms)
    {
      //System.out.println("("+m.length+", "+m[0].length+")");

      double[][] mSim = m.similarities;
      double[][] mDist = m.distances;
      if(aggregateSim == null)
      {
        aggregateSim = new double[mSim.length][mSim[0].length];
        aggregateDist = new double[mDist.length][mDist[0].length];
      }
      for(int i=0; i < mSim.length; i++)
      {
        for(int j=0; j < mSim[i].length; j++)
        {
          if(firstMatrix)
          {
            aggregateSim[i][j] = mSim[i][j];
            aggregateDist[i][j] = mDist[i][j];            
          }
          else
          {
            aggregateSim[i][j] += maxOrMin(aggregateSim[i][j], mSim[i][j], max);
            aggregateDist[i][j] += maxOrMin(aggregateDist[i][j], mDist[i][j], max);
          }
        }
      }
      firstMatrix = false;
    }
    return new DocMatrices(aggregateSim, aggregateDist);
  }

  private double maxOrMin(double d1, double d2, boolean max)
  {
    if(max)
      return Math.max(d1, d2);
    else
      return Math.min(d1, d2);
  }

  protected class DocMatrices
  {
    public double[][] similarities;
    public double[][] distances;
    public double[][] sentenceVectors;


    public DocMatrices(double[][] sentenceSimilarities, double[][] sentenceDistances)
    {
      this(sentenceSimilarities,sentenceDistances, null);
    }
    public DocMatrices(double[][] sentenceSimilarities, double[][] sentenceDistances, double[][] sentenceVectors)
    {
      this.similarities = sentenceSimilarities;
      this.distances = sentenceDistances;
      this.sentenceVectors = sentenceVectors;
    }
  }

  protected List<Map<String, Integer>> getSentencesAsBags(HashSet<String> stopwords, List<List<String>> sentencesLists, HashMap<String, String> wordClusterId, Collection<Flags> flags, HashMap<String, Double> idfs)
  {
    List<Map<String, Integer>> sentenceBagList = new ArrayList<Map<String, Integer>>();

    for(List<String> sentences: sentencesLists)
    {
      for(String sentence: sentences)
      {
        if (sentence.length() > 0)
        {
          HashMap<String, Integer> currSentence = new HashMap<String, Integer>();
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


            if(flags.contains(Flags.MOD_BIGRAMS) && prev != null)
            {
              String bigram = prev+" "+stemmed;

              if(currSentence.containsKey(bigram))
                currSentence.put(bigram, currSentence.get(bigram)+1);
              else
                currSentence.put(bigram, 1);


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

                if(currSentence.containsKey(bigram))
                  currSentence.put(clusterReplacedBigram, currSentence.get(bigram)+1);
                else
                  currSentence.put(clusterReplacedBigram, 1);
              }
            }
            //end, bigrams

            if(stopwords.contains(w) == false)
            {
              if(currSentence.containsKey(stemmed))
                currSentence.put(stemmed, currSentence.get(stemmed)+1);
              else
                currSentence.put(stemmed, 1);

              //also add cluster-replacement!
              if(wordClusterId.containsKey(stemmed))
              {
                String clusterId = wordClusterId.get(stemmed);
                if(currSentence.containsKey(clusterId))
                  currSentence.put(clusterId, currSentence.get(clusterId)+1);
                else
                  currSentence.put(clusterId, 1);
              }

              prev = stemmed;
            }
            else
              prev = w;

          }
          sentenceBagList.add(currSentence);
        }
      }
    }
    return sentenceBagList;
  }

  protected static HashMap<String, String> getWordClusters(String wordClusterFileName)
  {
    HashMap<String, String> wordClusterId = new HashMap<String, String>();
    if (wordClusterFileName != null)
    {
      try
      {
        BufferedReader reader = new BufferedReader(new FileReader(wordClusterFileName));
        String line = reader.readLine();
        int clusterIndex = 0;
        while (line != null)
        {
          String clusterId = "cluster-" + clusterIndex;
          String[] words = line.split(REGEX_SPACE);
          for(String word: words)
          {
            if (!wordClusterId.containsKey(word))
              wordClusterId.put(word, clusterId);
          }
          line = reader.readLine();
          clusterIndex++;
        }
        reader.close();
      }
      catch (Exception e)
      {
        e.printStackTrace();
        System.exit(1);
      }
    }
    return wordClusterId;
  }

  protected static HashSet<String> getStopwords(String stopwordsFilename)
  {
    HashSet<String> stopwords = new HashSet<String>();
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
    return stopwords;
  }

  protected int charactersLength(Collection<Integer> selected, List<String> sentences)
  {
    int bytes = 0;
    for(Integer i: selected)
    {
      bytes += sentences.get(i).length();
    }
    return bytes;
  }
  protected int wordsLength(Collection<Integer> selected, List<String> sentences)
  {
    int bytes = 0;
    for(Integer i: selected)
    {
      bytes += sentences.get(i).split(REGEX_NONWORD).length;
    }
    return bytes;
  }
  protected boolean summaryIsTooShort(Collection<Integer> selected, List<String> sentences, LengthUnit lengthUnit, int summarySize)
  {
    if(lengthUnit == LengthUnit.CHARACTERS)
      return charactersLength(selected, sentences) < summarySize;
    else if(lengthUnit == LengthUnit.WORDS)
      return wordsLength(selected, sentences) < summarySize;
    else
      return selected.size() < summarySize;
  }
  protected boolean summaryIsTooLong(Collection<Integer> selected, List<String> sentences, LengthUnit lengthUnit, int summarySize)
  {
    if(lengthUnit == LengthUnit.CHARACTERS)
      return charactersLength(selected, sentences) > summarySize;
      else if(lengthUnit == LengthUnit.WORDS)
        return wordsLength(selected, sentences) > summarySize;
        else
          return selected.size() > summarySize;
  }
  protected void saveLinTFIDFMatrix(double[][] similarities, String documentIdentifier) throws IOException
  {
    String pathname = System.getProperty("user.home")+"/tmp/Submod1Matrices/"+documentIdentifier;
    recursivelyCreateDirectory(new File(pathname));
    String filename = "/LinTFIDF.mat";
    BufferedWriter bw = new BufferedWriter(new FileWriter(new File(pathname+filename)));
    for(double[] row: similarities)
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
  
  public static void recursivelyCreateDirectory(File f) throws IOException
  {
    if(f != null && !f.exists())
    {
      recursivelyCreateDirectory(f.getParentFile());
      f.mkdir();
    }
  }
}
