# algoStageJuin23

All these algorithms have been developped by Martin AYMÉ (ENS de Lyon) during an internship at LGL-TPE, Universitée Lyon 1 Claude Bernard.
They were redacted in Python 3, utf-8.

The aim is to simplify data processing and complete a segregation of lithology using their chemistry. For this purpose :

- traitement_spectres : contains a list of functions that can process raw spectra (conversion from .mca to .csv, smoothening, ...)

- CourbesValidation : contains a list of fonctions that can format the results given by CloudCal calibrations and plot the validation curves to assess the calibration acuracy

- plotConcentration : contains a list of fonctions that can use the chemical data porvided by CloudCal quantification to plot concentration graphs, find clusters and quantify their accuracy
(clusterisation with DBSCAN, quantification with ajusted_rand_score and silhouette_score of the Scikit-learn module)

- compREEs : contains a list of fonctions that can plot and normalise spectra to make qualitative comparisons

The repository also contains a file with all the used CloudCal calibrations.