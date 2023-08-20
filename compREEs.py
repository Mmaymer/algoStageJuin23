# Python 3, UTF-8
# Martin AYMÉ
# ENS de Lyon, L3 Géosciences
# August 2023

## LIBRARIES

import matplotlib.pyplot as plt
import numpy as np

## MAIN

def graph(*,xMinLim=0.0,xMaxLim=41.0,Norm=True):

    VALUES = np.loadtxt("allSpectraEnergy.csv",skiprows=1,delimiter=",")
    file = open("allSpectraEnergy.csv")
    LOCATIONS = file.readline().strip().split(",")[1:]

    LABEL = getLabel(xMinLim,xMaxLim)


    x = VALUES[:,0]

    if Norm:
        yValuesNorm = normalize(VALUES)[:,1:]
    else:
        yValuesNorm = VALUES[:,1:]

    if xMaxLim-xMinLim != 41.0:
        startI,stopI = startStop(x,xMinLim,xMaxLim)
        yValuesNorm = yValuesNorm[startI:stopI,:]
        x = x[startI:stopI]

    for i in range(len(LOCATIONS)):
        plt.plot(x,yValuesNorm[:,i],label=LOCATIONS[i])

    yLim = list(plt.axis()[-2:])
    for j in range(len(LABEL[0])):
        plt.plot([LABEL[2][j]]*2,yLim,linestyle = ":",color="lightgrey",zorder=-1)
        plt.text(LABEL[2][j],yLim[0],LABEL[0][j]+f" ({LABEL[1][j]}, {round(LABEL[2][j],1)} keV)",rotation="vertical",size="small")

    #plt.legend()
    plt.xlabel("Energy (keV)")
    plt.ylabel("Intensity (cps/s)")
    plt.title("Spectres des radiolarites des différentes localités.\n(voltage = 50.0 kV, intensité = 60.0 µA, filtre 4, temps d'acquisition = 60 s, SciAps X-505)")
    plt.rcParams["figure.figsize"] = (17,9)
    plt.savefig(f"allSpectraPlot_{xMinLim}-{xMaxLim}keV_Norm-{Norm}.pdf",bbox_inches="tight")
    plt.show()




def normalize(VALUES):
    """Normalize all the spectra, i.e. calculate a coeficient for each spectrum on the portion between minE keV and maxE keV so that, when coeficient is applied, all spectra are superposed on the minE-maxE keV portion. Reference spectrum is CHE (index = 4)."""

    minE, maxE = 36.0,40.0 # select the gap of normalization

    start, stop = startStop(VALUES[:,0],minE,maxE)

    portions = VALUES[start:stop,1:]

    mean = np.array([sum(portions[:,i]) for i in range(len(portions[0,:]))])

    coefs = np.array([mean[4]/mean[i] for i in range(len(portions[0,:]))])

    normValues = np.concatenate((VALUES[:,0].reshape(-1,1),VALUES[:,1:]*coefs),axis=1)

    return normValues


def startStop(values,minE,maxE):
    """Get the indexes of start and stop, ex : index of 36 and 38 keV."""

    startE,stopE = minE,maxE

    startI = 0

    while values[startI] < startE:
        startI += 1

    stopI = startI

    while values[stopI] < stopE:
        stopI += 1

    return startI,stopI

def getLabel(xMinLim,xMaxLim):
    """Get the peak label to flag the different peaks observed, return a list with [[elementsList],[naturesOfPeakList],[energiesList]]"""

    file = open("peakLabel.csv")
    DATA = file.readlines()
    DATA = [Ele.strip().split(",") for Ele in DATA]
    filteredData = [Ele for Ele in DATA if float(Ele[2])>xMinLim and float(Ele[2])<xMaxLim]
    arrangedData = [[filteredData[i][j] for i in range(len(filteredData))] for j in range(3)]

    arrangedData[2] = [float(Ele) for Ele in arrangedData[2]]

    return arrangedData