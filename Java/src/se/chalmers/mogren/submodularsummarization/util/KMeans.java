package se.chalmers.mogren.submodularsummarization.util;

import java.util.Random;

public class KMeans
{
  private int[] cluster;
  private int numItems;
  private int numClusters;

  private void InitCluster()
  {
    for (int i = 0; i != numClusters && i != numItems; ++i)
      cluster[i] = i;
    Random rand = new Random(56432689);
    for (int i = numClusters; i != numItems; ++i)
      cluster[i] = rand.nextInt(numClusters);
  }

  private boolean UpdateCluster(float[][] Dist)
  {

    //compute centroid of a cluster as the element which minimizes its distance from the other elements in the same cluster
    int[] centroid = new int[numClusters];
    for (int k = 0; k != numClusters; ++k)
    {
      float min = Float.MAX_VALUE;
      for (int i = 0; i != numItems; ++i)
        if (cluster[i] == k)
        {
          float sum = 0.0f;
          for (int j = 0; j != numItems; ++j)
            if (i != j && cluster[j] == k)
              sum += Dist[i][j];
          if (sum < min)
          {
            min = sum;
            centroid[k] = i;
          }
        }
    }

    //assign each node to the cluster with the closest centroid
    int[] newcluster = new int[numItems];
    for (int i = 0; i != numItems; i++)
    {
      newcluster[i] = 0;
      for (int k = 1; k != numClusters; ++k)
        if (Dist[i][centroid[k]] < Dist[i][centroid[newcluster[i]]])
          newcluster[i] = k;
    }

    //look for any change in clustering and save modifications
    boolean isChanged = false;
    for (int i = 0; i != numItems; i++)
      if (newcluster[i] != cluster[i])
      {
        isChanged = true;
        cluster[i] = newcluster[i];
      }
    return isChanged;
  }

  public int[] Cluster(float[][] Dist)
  {
    this.cluster = new int[numItems];
    InitCluster();
    int counter = 0;
    while (UpdateCluster(Dist) && counter <= 100*numItems )
    {
      counter++;
    }
    return cluster;
  }

  public KMeans(int numItems, int numClusters)
  {
    this.numItems = numItems;
    this.numClusters = numClusters;
  }

  public void Show()
  {
    for (int k = 0; k != numClusters; ++k)
    {
      System.out.print("Cluster #" + k + " = { ");
      for (int j = 0; j != numItems; ++j)
        if (cluster[j] == k)
          System.out.print((j+1) + " ");
      System.out.println("}");
    }
  }
}
