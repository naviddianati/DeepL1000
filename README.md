# DeepL1000

Deep Learning models for predicting the mechanism of action of FDA approved drugs based on their differential gene expression profiles in multiple cellular contexts. The data covers about 1,400 drugs, each with signatures measured in at most 3 doses using the L1000 assay developed by the CMap group at the Broad Institute of MIT and Harvard. Labels are curated from literature.
This package provides a data preparation, preprocessing, train/valdiation and testing framework for the classification of the data using neural networks written in Tensorflow/Keras. Models presented here expect the input data for each drug/dose combination to be presented as a stack of 7 vectors representing the differential gene expression profiles of the compound in 7 cancer cell lines. 
