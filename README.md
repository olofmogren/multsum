# An implementation of Submodular Optimization for Extractive Multi-Document Summarization

Olof Mogren

(Or rather, two implementations, there is one in Java, and one in Python).


## Prerequisites

Both the Java and the Python version needs the CLUTO binary to run. It can be freely downloaded from:

http://glaros.dtc.umn.edu/gkhome/cluto/cluto/download

Both versions also rely on a stopwords-list (there is one distributed with ROUGE that can be used). In the Java version, this can be specified on the command line. See usage below.

At the time of writing, you have to specify the paths for these files in the source code. For the Java version, the paths are in SubSumBase.java, and for the Python version, the paths are in subsum.py and subsum\_clustering.py.


## Java version

The Java version is similar to the python version, but has been used more on the command line, and thus, it has more command line parameters supported.

#### Usage:

java SubSum.java --matrix-file <filename1> [--matrix-file <filename2> [--matrix-file <filename3> [...]]]
    --document-file file-or-directory
    --stopwords-file filename
    [--collection-for-idfs directory]
    [--word-clusters-file filename]
    [--nb-sentences summarysize]
    [--nb-words summarysize]
    [--nb-characters summarysize]
    [--lambda lambda]
    [--disable-bigrams]
    [--print-as-text]



## Python version

### Usage:

#### 1

subsum.py --s sentences-file [--m matrix-file1 [--m matrix-file2 [...]]]

Will summarize the document in sentences-file, using the sentence similarity matrices in the matrix-files. You can specify an arbitrary number of matrix-files, but in this mode, there has to be one sentences-file.

All input files are plain text files.

The sentence-file has to be split into sentences, one sentence per line.

The matrix-files must contain square matrices of the same dimension as the number of sentences in the sentences-file. Elements are separated by commas, rows separated by semi-colons:

a11, a12, a13;
a21, a22, a23;
a31, a32, a33;


#### 2

subsum.py sentences-file1[, sentences-file2[, sentences-file3[, ...]]]

The sentence-files has to be split into sentences, one sentence per line.

The summarizer will use all sentences as input and compute a summary that's representative for all input documents.


