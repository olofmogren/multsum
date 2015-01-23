using System;
using System.IO;
using System.Text.RegularExpressions;
using System.Collections.Generic;

class HLinJBilmesWithClusters
{
	static float L1(HashSet<int> S, int N, float[,] w, float alfa)
	{
		float res = 0.0f;
		for (int i = 0; i != N; ++i)
		{
			float sum = 0.0f, sumV = 0.0f;
			foreach (int j in S)
				sum += w[i, j];
			for (int k = 0; k != N; ++k)
				sumV += w[i, k];
			sumV *= alfa;
			res += sum < sumV ? sum : sumV;
		}
		return res;
	}
	static float R1(HashSet<int> S, int N, float[,] w, int[] clustering, int K)
	{
		float res = 0.0f;
		for (int k = 0; k != K; ++k)
		{
			float sum = 0.0f;
			foreach (int j in S)
				if (clustering[j] == k)
				{
					float sumV = 0.0f;
					for (int i = 0; i != N; ++i)
						sumV += w[i, j];
					sum += sumV / N;
				}
			res += (float)Math.Sqrt(sum);
		}
		return res;
	}
	static Regex regex = new Regex(@"\W+", RegexOptions.Compiled);
	static void Main (string[] args)
	{
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

		Dictionary<string, string> wordClusterId = new Dictionary<string, int>();
		//Dictionary<string, List<string>> wordClusterFromId = new Dictionary<int, List<string>>();
		try
		{
			StreamReader reader = new StreamReader (args [3]);
			string line = reader.ReadLine ();
			int clusterIndex = 0;
			while (line != null)
			{
				string clusterId = "cluster-"+clusterIndex;
				string[] words = regex.Split(line);
				foreach(string word in words)
				{
					wordClusterId.Add(word, clusterId);
					//if(!wordClusterFromId.ContainsKey(clusterId))
					//	wordClusterFromId[clusterId] = new List<string>();
					//wordClusterFromId[clusterId].Add(word);
				}
				line = reader.ReadLine ();
				clusterIndex++;
			}
		}
		catch (Exception e)
		{
			Console.Error.WriteLine (e);
			Environment.Exit (1);
		}

		try
		{
			Dictionary<string, float[]> document = new Dictionary<string, float[]>();
			List<Dictionary<string, uint>> sentences = new List<Dictionary<string, uint>>();
			StreamReader reader = new StreamReader(args[1]);
			string line = reader.ReadLine();
			while (line != null)
			{
				if (line.Length > 0)
				{
					Dictionary<string, uint> currsentence = new Dictionary<string, uint>();
					string[] words = regex.Split(line);
					foreach (string w in words)
						if (w.Length != 0 && stopwords.Contains(w.ToLower()) == false)
						{
							string stemmed = w.ToLower().ToPorterStem();

							// olof: replace words with their cluster id:
						    if(wordClusterId.ContainsKey(stemmed))
							{
								Console.WriteLine("replacing word "+stemmed+" with its cluster id: "+wordClusterId[stemmed]);
								stemmed = wordClusterId[stemmed];
							}

							if (document.ContainsKey(stemmed))
								document[stemmed][0]++;
							else
								document.Add(stemmed, new float[2] { 1.0f, 0.0f });
							if (currsentence.ContainsKey(stemmed))
								currsentence[stemmed]++;
							else
							{
								currsentence.Add(stemmed, 1);
								document[stemmed][1]++;
							}
						}
					sentences.Add(currsentence);
				}
				line = reader.ReadLine();
			}
			reader.Close();

			float[] tfidfden = new float[sentences.Count];
			float[] idfden = new float[sentences.Count];
			for (int i = 0; i != sentences.Count; ++i)
			{
				tfidfden[i] = idfden[i] = 0.0f;
				foreach (KeyValuePair<string, uint> pair in sentences[i])
				{
					float idf = document[pair.Key][1];
					tfidfden[i] += pair.Value * pair.Value * idf * idf;
					idfden[i] += idf * idf;
				}
				tfidfden[i] = (float)Math.Sqrt(tfidfden[i]);
				idfden[i] = (float)Math.Sqrt(idfden[i]);
			}

			float[,] tfidfsim = new float[sentences.Count, sentences.Count];
			float[,] idfdist = new float[sentences.Count, sentences.Count];
			for (int i = 0; i != sentences.Count; ++i)
			{
				tfidfsim[i, i] = 1.0f;
				idfdist[i, i] = 0.0f;
				for (int j = i + 1; j != sentences.Count; ++j)
				{
					float tfidfnum = 0.0f, idfnum = 0.0f;
					foreach (KeyValuePair<string, uint> pair in sentences[i])
						if (sentences[j].ContainsKey(pair.Key))
						{
							float idf = document[pair.Key][1];
							tfidfnum += pair.Value * idf * sentences[j][pair.Key] * idf;
							idfnum += idf * idf;
						}
					tfidfsim[i, j] = tfidfsim[j, i] = tfidfnum / (tfidfden[i] * tfidfden[j]);
					idfdist[i, j] = idfdist[j, i] = 1.0f - idfnum / (idfden[i] * idfden[j]);
				}
			}

			int K = (int)(0.2f * (float)sentences.Count + 0.5f);
			const float lambda = 6.0f;
			const float alfa = 0.75f;
			int summarysize = int.Parse(args[2]);

			KMeans kmeans = new KMeans(sentences.Count, K);
			int[] clustering = kmeans.Cluster(idfdist);
//			kmeans.Show();

			HashSet<int> selected = new HashSet<int>();
			while (selected.Count != summarysize && selected.Count != sentences.Count)
			{
				float max = 0.0f;
				int argmax = sentences.Count;
				for (int i = 0; i != sentences.Count; ++i)
					if (selected.Contains(i) == false)
					{
						selected.Add(i);
						float curr = L1(selected, sentences.Count, tfidfsim, alfa) + lambda * R1(selected, sentences.Count, tfidfsim, clustering, K);
						if (curr > max)
						{
							argmax = i;
							max = curr;
						}
						selected.Remove(i);
					}
				if (argmax != sentences.Count)
				{
					selected.Add(argmax);
					Console.WriteLine(argmax + 1);
				}
				else break;
			}
		}
		catch (Exception e)
		{
			Console.Error.WriteLine(e);
			Environment.Exit(2);
		}
	}
}
