========================
 Spectral Clustering
 Ingo Bürk
 Bachelor Thesis
 University of Stuttgart
 2012
========================


==================================
=========
CONTENTS:
=========

1. General Information
2. Requirements
3. Installation
4. Usage
5. Dataset specifications
6. Similarity graph specifications
7. Clustered data specifications
8. Save plot specifications
9. References

==================================



1. GENERAL INFORMATION

This software was written as part of a Bachelor Thesis. It represents an implementation of
spectral clustering algorithms, including a complete graphical user interface (GUI). All rights
belong to the author.



2. REQUIREMENTS

To run this software, you need to have the following components installed:
- Mathworks MATLAB
- Mathworks Statistics Toolbox
- recommended: export_fig configured to save to pdf files (Mathworks File Exchange ID #23629)



3. INSTALLATION

This software doesn't require any installation. Just drop the files into a folder.



4. USAGE

To run the software, run the file 'Main.m'. The script will take care of all the rest and start a graphical user interface. The basic usage is as follows:
- Load a dataset (see dataset specifications)
- Choose similarity graph parameters and create similarity graph
- Choose clustering parameters and cluster data
- Save clustered data or plot them

For more detailed explanations, please refer to the tutorial that comes with this thesis. However, the user interface should, for the most part, be easy to understand.



5. DATASET SPECIFICATIONS

Datasets compatible with this software need to be in a comma-separated value form (csv-Files). Each row represents a data point, the columns represent the dimensions. The software differentiates two types of files:

-- Labeled Files (*.csv)
Files ending in *.csv are assumed to be labeled, i. e. containing labels in the first column that will be ignored when the dataset is loaded.

-- Non-Labeled Files (*.nld)
Files ending in *.nld are assumed to be unlabeled and all columns will be loaded.



6. SIMILARITY GRAPH SPECIFICATIONS

Similarity graphs are stored in the internal file mat-File format. Please refer to the MATLAB Help on 'matfile' for information on how to read or create such files. The files created by this software hold the following information, where fileObj is the mat-File object:
- fileObj.SimGraph: Contains the sparse similarity graph
- fileObj.SimGraphType: Type with which the graph was generated (1 = Full, 2 = Normal kNN, 3 = Mutual kNN, 4 = Epsilon)
- fileObj.Neighbors: Number of Neighbors
- fileObj.Eps: Epsilon value
- fileObj.Sigma: Sigma value
- fileObj.Components: Number of connected components (0 if it couldn't be determined)

Please keep in mind that, depending on the type that was chosen, not all information are relevant to the graph stored in the file.



7. CLUSTERED DATA SPECIFICATIONS

Clustered data are saved in the same format describe above for the datasets, where an additional column is placed on the left (i. e. it's the first column), holding the corresponding cluster index.



8. SAVE PLOT SPECIFICATIONS

Plots saved through the graphical user interface will be saved into a pdf-File, using the script 'export_fig'. If this script cannot be found, saving plots will only be possible via builtin functions. Please keep in mind, that the builtin save functions of MATLAB mean a loss of quality, while saving via 'export_fig' allows the plots to be saved in a vector format without any quality loss.



9. REFERENCES

This software makes use of the following submissions to the MathWorks File Exchange:
- 'relativepath' by Jochen Lenz (ID #3858)
- 'export_fig' by Oliver Woodford (ID #23629)