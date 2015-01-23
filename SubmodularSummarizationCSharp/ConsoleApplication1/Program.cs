using System;
using System.IO;
using System.Text.RegularExpressions;
using System.Collections.Generic;


class HLinJBilmes
{
  static float L1 (HashSet<int> S, int N, float[,] w, float alfa)
  {
    float res = 0.0f;
    for (int i = 0; i != N; ++i) {
      float sum = 0.0f, sumV = 0.0f;
      foreach (int j in S)
        sum += w [i, j];
      for (int k = 0; k != N; ++k)
        sumV += w [i, k];
      sumV *= alfa;
      res += sum < sumV ? sum : sumV;
    }
    return res;
  }

  static float R1 (HashSet<int> S, int N, float[,] w, int[] clustering, int K)
  {
    float res = 0.0f;
    for (int k = 0; k != K; ++k) {
      float sum = 0.0f;
      foreach (int j in S)
        if (clustering [j] == k) {
          float sumV = 0.0f;
          for (int i = 0; i != N; ++i)
            sumV += w [i, j];
          sum += sumV / N;
        }
      res += (float)Math.Sqrt (sum);
    }
    return res;
  }

  static Regex regex = new Regex (@"\W+", RegexOptions.Compiled);
  static Regex regexSpaceComma = new Regex (@"[\s,]+", RegexOptions.Compiled);

  static void Main (string[] args)
  {
    if (args.Length <= 0)
    {
      Console.WriteLine("Usage: <command> stopwords-filename document-filename summarysize [word-cluster-file|- [external-similarity-matrix|- [lambda]]]\n(word-cluster-file and external-similarity-matrix can be replaced by a dash (-) as placeholder to supply no such file.");
      Environment.Exit (1);
    }
    //Some initializations:
    float[,] sentenceSimilarities = null;
    int[] clustering = null;
    int K = -1;
    float lambda = 6.0f;
    if (args.Length >= 6 && args[5] != "-")
    {
      lambda = int.Parse(args[5]);
    }
    const float alfa = 0.75f;
    int summarysize = int.Parse (args [2]);


    if (args.Length < 5 || args [4] == "-")
    {
      //Console.WriteLine ("Using internal TFIDF-based sentence similaritites.");
    var stopwords = new List<string> ();
    try {
      StreamReader reader = new StreamReader (args [0]);
      string line = reader.ReadLine ();
      while (line != null) {
        stopwords.Add (line);
        line = reader.ReadLine ();
      }
    } catch (Exception e) {
      Console.Error.WriteLine (e);
      Environment.Exit (1);
    }

    Dictionary<string, string> wordClusterId = new Dictionary<string, string> ();

    if (args.Length >= 4 && args [3] != "-") {
      try {
        StreamReader reader = new StreamReader (args [3]);
        string line = reader.ReadLine ();
        int clusterIndex = 0;
        while (line != null) {
          string clusterId = "cluster-" + clusterIndex;
          string[] words = regex.Split (line);
          foreach (string word in words) {
            if (!wordClusterId.ContainsKey (word))
              wordClusterId.Add (word, clusterId);
            //if(!wordClusterFromId.ContainsKey(clusterId))
            //  wordClusterFromId[clusterId] = new List<string>();
            //wordClusterFromId[clusterId].Add(word);
          }
          line = reader.ReadLine ();
          clusterIndex++;
        }
      } catch (Exception e) {
        Console.Error.WriteLine (e);
        Environment.Exit (1);
      }
    }

    try {
      Dictionary<string, float[]> document = new Dictionary<string, float[]> ();
      List<Dictionary<string, uint>> sentences = new List<Dictionary<string, uint>> ();
      StreamReader reader = new StreamReader (args [1]);
      string line = reader.ReadLine ();
      while (line != null) {
        if (line.Length > 0) {
          Dictionary<string, uint> currsentence = new Dictionary<string, uint> ();
          string[] words = regex.Split (line);
          foreach (string w in words)
            {
              //Console.Write(w+" ");
            if (w.Length != 0 && stopwords.Contains (w.ToLower ()) == false) {
              string stemmed = w.ToLower ().ToPorterStem ();

              // olof: replace words with their cluster id:
              if (wordClusterId.ContainsKey (stemmed)) {
                //Console.WriteLine("replacing word "+stemmed+" with its cluster id: "+wordClusterId[stemmed]);
                stemmed = wordClusterId [stemmed];
              }

              if (document.ContainsKey (stemmed))
                document [stemmed] [0]++;
              else
                document.Add (stemmed, new float[2] { 1.0f, 0.0f });
              if (currsentence.ContainsKey (stemmed))
                currsentence [stemmed]++;
              else {
                currsentence.Add (stemmed, 1);
                document [stemmed] [1]++;
              }
            }
            }
            //Console.WriteLine();
          sentences.Add (currsentence);
        }
        line = reader.ReadLine ();
      }
      reader.Close ();

      float[] tfidfden = new float[sentences.Count];
      float[] idfden = new float[sentences.Count];
      for (int i = 0; i != sentences.Count; ++i) {
        tfidfden [i] = idfden [i] = 0.0f;
        foreach (KeyValuePair<string, uint> pair in sentences[i]) {
          float idf = document [pair.Key] [1];
          tfidfden [i] += pair.Value * pair.Value * idf * idf;
          idfden [i] += idf * idf;
        }
        tfidfden [i] = (float)Math.Sqrt (tfidfden [i]);
        idfden [i] = (float)Math.Sqrt (idfden [i]);
      }

      float[,] tfidfsim = new float[sentences.Count, sentences.Count];
      float[,] idfdist = new float[sentences.Count, sentences.Count];
      for (int i = 0; i != sentences.Count; ++i) {
        tfidfsim [i, i] = 1.0f;
        idfdist [i, i] = 0.0f;
        for (int j = i + 1; j != sentences.Count; ++j) {
          float tfidfnum = 0.0f, idfnum = 0.0f;
          foreach (KeyValuePair<string, uint> pair in sentences[i])
            if (sentences [j].ContainsKey (pair.Key)) {
              float idf = document [pair.Key] [1];
              tfidfnum += pair.Value * idf * sentences [j] [pair.Key] * idf;
              idfnum += idf * idf;
            }
          //tfidfsim [i, j] = tfidfsim [j, i] = tfidfnum / (tfidfden [i] * tfidfden [j]);
          //idfdist [i, j] = idfdist [j, i] = 1.0f - idfnum / (idfden [i] * idfden [j]);

          tfidfsim[i, j] = tfidfsim[j, i] = (tfidfden[i]==0 || tfidfden[j]==0) ? 0.0f : tfidfnum / (tfidfden[i] * tfidfden[j]);
          idfdist[i, j] = idfdist[j, i] = (idfden[i]==0 || idfden[j]==0) ? 1.0f : 1.0f - idfnum / (idfden[i] * idfden[j]);
          //Console.Write("("+tfidfsim[j,i]+","+idfdist[j,i]+")");
        }
          //Console.WriteLine();
      }

      /*for(int i=0; i < sentences.Count; i++)
        {
          for(int j=0; j<sentences.Count; j++)
            Console.Out.Write(idfdist[i,j]+" ");
          Console.Out.WriteLine();
        }*/

        
      K = (int)(0.2f * (float)sentences.Count + 0.5f);
      KMeans kmeans = new KMeans (sentences.Count, K);
      clustering = kmeans.Cluster (idfdist);
      //kmeans.Show ();

      sentenceSimilarities = tfidfsim;

        
      } catch (Exception e) {
        Console.Error.WriteLine (e);
        Environment.Exit (2);
      }
    }
    else
    {
      //Console.Out.WriteLine ("Using kernel matrix file: "+args[4]+".");
        //if(args[4] == "")
        //  foreach(string w in args)
        //    Console.Out.WriteLine(w);

        try {
          StreamReader simReader = new StreamReader (args [4]);
          string simLine = simReader.ReadLine ();
          int lineNumber = 0;
          while (simLine != null)
          {
            string[] words = regexSpaceComma.Split (simLine);
            if(sentenceSimilarities == null)
            {
              sentenceSimilarities = new float[words.Length,words.Length];
            }
            else if (words.Length != sentenceSimilarities.GetLength(0))
            {
              Console.Error.WriteLine ("Mismatch: found " + words.Length + " similaritities (cols), but there are " + sentenceSimilarities.GetLength(0) + " sentences.");
              Environment.Exit (1);
            }

            int wordNumber = 0;
            foreach (string word in words) {
              //      Console.Out.WriteLine(word);
              float value;
              float.TryParse (word, out value);
              //      Console.Out.WriteLine(value);
              sentenceSimilarities [lineNumber, wordNumber] = value;
              wordNumber++;
            }
            simLine = simReader.ReadLine ();
            lineNumber++;
          }
            
          if (lineNumber != sentenceSimilarities.GetLength(0)) {
            Console.Error.WriteLine ("Mismatch: found " + lineNumber + " similaritities (lines), but there are " + sentenceSimilarities.GetLength(0) + " sentences.");
            Environment.Exit (1);
          }
        } catch (Exception e) {
          Console.Error.WriteLine (e);
          Environment.Exit (1);
        }

      float[,] dissimilarities = new float[sentenceSimilarities.GetLength(0), sentenceSimilarities.GetLength(0)];
      for (int i=0; i < sentenceSimilarities.GetLength(0); i++) {
        for (int j=0; j<sentenceSimilarities.GetLength(0); j++) {
            dissimilarities [i, j] = 1.0f - sentenceSimilarities [i, j];
            /*if(sentenceSimilarities[i,j] == 0.0)
        dissimilarities [i, j] = float.PositiveInfinity;
      else
              dissimilarities [i, j] = (1.0f / sentenceSimilarities [i, j]) - 1.0f;*/
          }
        }

        /*for (int i=0; i < sentences.Count; i++) {
          for (int j=0; j<sentences.Count; j++)
            //Console.Out.Write ("("+sentenceSimilarities [i, j] + "," + dissimilarities[i,j]+","+idfdist[i,j]+") ");
            //Console.Out.Write ("("+sentenceSimilarities [i, j] + "," + dissimilarities[i,j]+","+idfdist[i,j]+") ");
            Console.Out.Write ("("+sentenceSimilarities [i, j]+") ");
          Console.Out.WriteLine ();
        }*/

      
      K = (int)(0.2f * (float)sentenceSimilarities.GetLength(0) + 0.5f);
      KMeans kmeans = new KMeans (sentenceSimilarities.GetLength(0), K);
        clustering = kmeans.Cluster (dissimilarities);
        //kmeans.Show ();
      }
      
      //for(int i = 0; i < sentences.Count; i++)
      //{
      //  for(int j = 0; j < sentences.Count; j++)
      //    Console.Out.Write (sentenceSimilarities[i,j]+" ");
      //  Console.Out.WriteLine("");
      //}

      HashSet<int> selected = new HashSet<int> ();
    while (selected.Count != summarysize && selected.Count != sentenceSimilarities.GetLength(0)) {
        float max = 0.0f;
      int argmax = sentenceSimilarities.GetLength(0);
      for (int i = 0; i != sentenceSimilarities.GetLength(0); ++i)
          if (selected.Contains (i) == false) {
            selected.Add (i);
        float curr = L1 (selected, sentenceSimilarities.GetLength(0), sentenceSimilarities, alfa) + lambda * R1 (selected, sentenceSimilarities.GetLength(0), sentenceSimilarities, clustering, K);
            //Console.Out.WriteLine("i= "+i+",curr "+curr+", max= "+max);
            //Console.Out.WriteLine("L1= "+L1 (selected, sentences.Count, sentenceSimilarities, alfa));
            //Console.Out.WriteLine("R1= "+R1 (selected, sentences.Count, sentenceSimilarities, clustering, K));
            if (curr > max) {
              argmax = i;
              max = curr;
            }
            selected.Remove (i);
          }
      if (argmax != sentenceSimilarities.GetLength(0)) {
          //Console.Out.WriteLine("found argmax "+(argmax+1));
          selected.Add (argmax);
          Console.WriteLine (argmax + 1);
        } else
          break;
      }
  }
}
