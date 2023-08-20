# Martin AYMÉ (martin.ayme@ens-lyon.fr)
# ENS de Lyon
# Python, UTF-8
# July 2023

## Libraries

import courbesValidation as cVal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import DBSCAN
from datetime import datetime as dt

## Main

def plotConcent(xNumEle,yNumEle,*,xDenEle=[1],yDenEle=[1]):
    """Plot a graph with the concentrations of xNumEle/xDenEle on x-axis and yNumEle/yDenEle on y-axis, x and yDenEle can be {1} or element(s)"""

    # ATTENTION :
    # For the whole fonction, the samples order is, and must be, conserved
    # according Voreppe → Chenaillet → Corse, else, lines 51: won't work correctly

    mesureVoreppe = np.array(cVal.loadData("mesureRadioVoreppe.csv"))
    mesureChenaillet = np.array(cVal.loadData("mesuresRadioChenaillet.csv"))
    mesureCorse = np.array(cVal.loadData("mesuresRadioCorse.csv"))
    mesureForville = np.array(cVal.loadData("mesuresRadioForville.csv"))

    allElements = list(mesureCorse[0][1:]) # get the list of all elements quantified
    allSamples = list(mesureVoreppe[1:,0]) + list(mesureChenaillet[1:,0]) + list(mesureCorse[1:,0]) + list(mesureForville[1:,0])
    allSamplesLoc = ["VOR"]*len(mesureVoreppe[1:,0]) + ["CHE"]*len(mesureChenaillet[1:,0]) + ["COR"]*len(mesureCorse[1:,0]) + ["FOR"]*len(mesureForville[1:,0])

    requestedEle = [xNumEle,xDenEle,yNumEle,yDenEle]
    requestedEleSolo = [] # list without repetition of all elements requested

    for part in requestedEle:
        for i in range(len(part)):
            if part[i] not in requestedEleSolo and part[i] in allElements:
                # append if not already appended and present in allElements (i.e. is not "1")
                requestedEleSolo.append(part[i])

    eleIndex = [] # index of requested elmts in allElements, also true in all mesureXXXX
    compoEle = [0]*len(requestedEleSolo) # [ [ element1Voreppe,elmt1Chenaillet,elmt1Corse ],[same for elmt 2] ]

    for i in range(len(requestedEleSolo)):
        eleIndex.append(allElements.index(requestedEleSolo[i]))

        compoEleI = [float(mesure) for mesure in mesureVoreppe[1:,eleIndex[i]+1]]
        compoEleI += [float(mesure) for mesure in mesureChenaillet[1:,eleIndex[i]+1]]
        compoEleI += [float(mesure) for mesure in mesureCorse[1:,eleIndex[i]+1]]
        compoEleI += [float(mesure) for mesure in mesureForville[1:,eleIndex[i]+1]]

        compoEle[i] = np.array(compoEleI)
        # for each elmt, add an array with the composition of the location

    # x-axis
    nameXAxis = ""

    # x-axis num
    xNum = np.array([0.0]*len(allSamples))
    for elmt in xNumEle:
        index = requestedEleSolo.index(elmt) # get the index of elmt in requestedEleSolo
        xNum += compoEle[index]

    if len(xNumEle) == 1:
        nameXAxis = f"[{xNumEle[0]}]"
    else:
        nameXAxis = f"[{xNumEle[0]}]"
        for i in range(1,len(xNumEle)):
            nameXAxis += f"+[{xNumEle[i]}]"

    # x-axis den
    xDen = np.array([0.0]*len(allSamples))

    if xDenEle == [1]:
        xDen = np.array([1]*len(allSamples))

    else:
        for elmt in xDenEle:
            index = requestedEleSolo.index(elmt)
            xDen += compoEle[index]

        if len(xDenEle) == 1:
            nameXAxis = f" / [{xDenEle[0]}]"
        else:
            nameXAxis = f" / [{xDenEle[0]}]"
            for i in range(1,len(xDenEle)):
                nameXAxis += f"+[{xDenEle[i]}]"

    X = list(xNum/xDen)

    # y-axis
    nameYAxis = ""

    # y-axis num
    yNum = np.array([0.0]*len(allSamples))
    for elmt in yNumEle:
        index = requestedEleSolo.index(elmt)
        yNum += compoEle[index]

    if len(yNumEle) == 1:
        nameYAxis = f"[{yNumEle[0]}]"
    else:
        nameYAxis = f"[{yNumEle[0]}]"
        for i in range(1,len(yNumEle)):
            nameYAxis += f"+[{yNumEle[i]}]"

    # y-axis den
    yDen = np.array([0.0]*len(allSamples))

    if yDenEle == [1]:
        yDen = np.array([1]*len(allSamples))

    else:
        for elmt in yDenEle:
            index = requestedEleSolo.index(elmt)
            yDen += compoEle[index]

        if len(yDenEle) == 1:
            nameYAxis = f" / [{yDenEle[0]}]"
        else:
            nameYAxis = f" / [{yDenEle[0]}]"
            for i in range(1,len(yDenEle)):
                nameYAxis += f"+[{yDenEle[i]}]"

    Y = list(yNum/yDen)
    #print("XY >>",X,Y)

    markersList = np.array(["."]*len(mesureVoreppe[1:,0]) + ["x"]*len(mesureChenaillet[1:,0]) + ["+"]*len(mesureCorse[1:,0]) + ["1"]*len(mesureForville[1:,0]))
    colorsList = ["blue"]*len(mesureVoreppe[1:,0]) + ["orange"]*len(mesureChenaillet[1:,0]) +["lightgreen"]*len(mesureCorse[1:,0]) + ["magenta"]*len(mesureForville[1:,0])



    for i in range(len(allSamples)):

        plt.scatter(X[i:i+1],Y[i:i+1],marker = markersList[i], c = colorsList[i], label = allSamples[i])

    legendElements = [plt.Line2D([],[],marker=".",markerfacecolor="b",markersize=10,mew = 0,label="Voreppe",ls=""),plt.Line2D([],[],marker="x",markerfacecolor="orange",markersize=7,mec = "orange",label="Chenaillet",ls=""),plt.Line2D([],[],marker="+",markerfacecolor="lightgreen",markersize=8,mec = "lightgreen",label="Corse",ls=""),plt.Line2D([],[],marker="1",markerfacecolor="magenta",markersize=8,mec = "magenta",label="Forville",ls="")]
    plt.legend(handles=legendElements)
    plt.xlabel(nameXAxis+" (wt. %)")
    plt.ylabel(nameYAxis+" (wt. %)")
    plt.show()


## Clustering

def blob():
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=750, centers=centers, cluster_std=0.4, random_state=0
    )

    X = StandardScaler().fit_transform(X)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    return X

def makeClusters(X,*,epsilon=0.3,minSamples=10):
    db = DBSCAN(eps=epsilon, min_samples=minSamples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    #print("Estimated number of clusters: %d" % n_clusters_)
    #print("Estimated number of noise points: %d" % n_noise_)

    return db, n_clusters_, n_noise_



def cluster(xNumEle,yNumEle,*,xDenEle=[1],yDenEle=[1],epsilon=1,minSamples=2, plot=True,scaled=False,saveFig=False):
    """Plot a graph with the concentrations of xNumEle/xDenEle on x-axis and yNumEle/yDenEle on y-axis, x and yDenEle can be {1} or element(s) and dissociate by color the points in clusters if DBSCAN can find any

    - epsilon (float, default : 1) is the factor added in the clusterisation process, it tries to find clusters a "diameter" or a spread of epsilon*minimumOfStdDeviationOfXAndYLists
    - minSamples (int, default : 2) is the minimal size for a cluster, default is 2 since the smallest predicted cluster's size is 2
    - plot (bool, dflt : True) : if True, plots the graph
    - scaled (bool, dflt : False) : if True, gives a normed graph
    - saveFig (bool, dflt : False) : if True, save se graph"""

    # ATTENTION :
    # For the whole fonction, the samples order is, and must be, conserved
    # as Voreppe → Chenaillet → Corse → Forville → Philippe, else, lines 217
    # won't work correctly

    mesureVoreppe = np.array(cVal.loadData("mesureRadioVoreppeAvg.csv"))
    mesureChenaillet = np.array(cVal.loadData("mesuresRadioChenailletAvg.csv"))
    mesureCorse = np.array(cVal.loadData("mesuresRadioCorse.csv"))
    mesureForville = np.array(cVal.loadData("mesuresRadioForville.csv"))
    mesurePhilippe = np.array(cVal.loadData("mesuresRadioPhilippe.csv"))

    allElements = list(mesureCorse[0][1:]) # get the list of all elements quantified
    allSamples = list(mesureVoreppe[1:,0]) + list(mesureChenaillet[1:,0]) + list(mesureCorse[1:,0]) + list(mesureForville[1:,0]) + list(mesurePhilippe[1:,0])
    allSamplesLoc = ["VOR"]*len(mesureVoreppe[1:,0]) + ["CHE"]*len(mesureChenaillet[1:,0]) + ["COR"]*len(mesureCorse[1:,0]) + ["FOR"]*len(mesureForville[1:,0]) + ["RAD"]*len(mesurePhilippe[1:,0])
    labelsTrue = [0]*len(mesureVoreppe[1:,0]) + [1]*len(mesureChenaillet[1:,0]) + [2]*len(mesureCorse[1:,0]) + [3]*len(mesureForville[1:,0]) + [4]*len(mesurePhilippe[1:,0])
    allPossibLabelOrder = allLabelPossib # get all the possible arrannements of label

    allLabel = [[]]*len(allPossibLabelOrder)

    for i in range(len(allPossibLabelOrder)):# create all the reference clusters to be compared with DBSCAN result (labelsPred when modified)

        possibLabel = allPossibLabelOrder[i]

        allLabel[i] = [possibLabel[0]]*len(mesureVoreppe[1:,0]) + [possibLabel[1]]*len(mesureChenaillet[1:,0]) + [-1]*len(mesureCorse[1:,0]) + [possibLabel[2]]*len(mesureForville[1:,0]) + [possibLabel[3]]*len(mesurePhilippe[1:,0])

    requestedEle = [xNumEle,xDenEle,yNumEle,yDenEle]
    requestedEleSolo = [] # list without repetition of all elements requested

    for part in requestedEle:
        for i in range(len(part)):
            if part[i] not in requestedEleSolo and part[i] in allElements:
                # append if not already appended and present in allElements (i.e. is not "1")
                requestedEleSolo.append(part[i])

    eleIndex = [] # index of requested elmts in allElements, also true in all mesureXXXX
    compoEle = [0]*len(requestedEleSolo) # [ [ element1Voreppe,elmt1Chenaillet,elmt1Corse ],[same for elmt 2] ]

    for i in range(len(requestedEleSolo)):
        eleIndex.append(allElements.index(requestedEleSolo[i]))

        compoEleI = [float(mesure) for mesure in mesureVoreppe[1:,eleIndex[i]+1]]
        compoEleI += [float(mesure) for mesure in mesureChenaillet[1:,eleIndex[i]+1]]
        compoEleI += [float(mesure) for mesure in mesureCorse[1:,eleIndex[i]+1]]
        compoEleI += [float(mesure) for mesure in mesureForville[1:,eleIndex[i]+1]]
        compoEleI += [float(mesure) for mesure in mesurePhilippe[1:,eleIndex[i]+1]]

        compoEle[i] = np.array(compoEleI)
        # for each elmt, add an array with the composition of the location

    # x-axis
    nameXAxis = ""

    # x-axis num
    xNum = np.array([0.0]*len(allSamples))
    for elmt in xNumEle:
        index = requestedEleSolo.index(elmt) # get the index of elmt in requestedEleSolo
        xNum += compoEle[index] # for each element, add the concentration in xNum

    if len(xNumEle) == 1:
        nameXAxis = f"[{xNumEle[0]}]"
    else:
        nameXAxis = f"[{xNumEle[0]}]" # create the name of the x-axis
        for i in range(1,len(xNumEle)):
            nameXAxis += f"+[{xNumEle[i]}]"

    # x-axis den
    xDen = np.array([0.0]*len(allSamples))

    if xDenEle == [1]:
        xDen = np.array([1]*len(allSamples)) # if 1, no concentration on the den

    else:
        for elmt in xDenEle:
            index = requestedEleSolo.index(elmt)
            xDen += compoEle[index] # else, same than xNum

        if len(xDenEle) == 1:
            nameXAxis = f" / [{xDenEle[0]}]"
        else:
            nameXAxis = f" / [{xDenEle[0]}]"
            for i in range(1,len(xDenEle)):
                nameXAxis += f"+[{xDenEle[i]}]"

    X = list(xNum/xDen) # divide xNum by xDen to generate the final X list that will be plotted

    # y-axis
    nameYAxis = "" # same for the y-axis

    # y-axis num
    yNum = np.array([0.0]*len(allSamples))
    for elmt in yNumEle:
        index = requestedEleSolo.index(elmt)
        yNum += compoEle[index]

    if len(yNumEle) == 1:
        nameYAxis = f"[{yNumEle[0]}]"
    else:
        nameYAxis = f"[{yNumEle[0]}]"
        for i in range(1,len(yNumEle)):
            nameYAxis += f"+[{yNumEle[i]}]"

    # y-axis den
    yDen = np.array([0.0]*len(allSamples))

    if yDenEle == [1]:
        yDen = np.array([1]*len(allSamples))

    else:
        for elmt in yDenEle:
            index = requestedEleSolo.index(elmt)
            yDen += compoEle[index]

        if len(yDenEle) == 1:
            nameYAxis = f" / [{yDenEle[0]}]"
        else:
            nameYAxis = f" / [{yDenEle[0]}]"
            for i in range(1,len(yDenEle)):
                nameYAxis += f"+[{yDenEle[i]}]"

    Y = list(yNum/yDen)

    xStdDev = [0]*len(set(labelsTrue)) # get the std deviation for the x-list by location
    yStdDev = [0]*len(xStdDev) # same for the y-list
    for i in range(len(labelsTrue)):
        stdDev = np.std([X[j] for j in range(len(X)) if labelsTrue[j]==labelsTrue[i]])
        xStdDev[labelsTrue[i]] = stdDev

        stdDev = np.std([Y[j] for j in range(len(Y)) if labelsTrue[j]==labelsTrue[i]])
        yStdDev[labelsTrue[i]] = stdDev

    maxStdDev = round(max(max(xStdDev),max(yStdDev)),3) # get the max of std deviations
    #maxStdDev = round(max(sum(xStdDev)/len(xStdDev),sum(yStdDev)/len(yStdDev)),3) # get the max of std deviations

    points = np.zeros([len(X),2])
    for i in range(len(X)):
        points[i,:] = [X[i],Y[i]] # make an array [[X1,Y1],[X2,Y2], ...] for the "makeClusters" function

    markersList = np.array(["."]*len(mesureVoreppe[1:,0]) + ["x"]*len(mesureChenaillet[1:,0]) + ["+"]*len(mesureCorse[1:,0]) + ["1"]*len(mesureForville[1:,0]) + ["3"]*len(mesurePhilippe[1:,0]))

    db,n_clusters,n_noise = makeClusters(points,epsilon=epsilon*maxStdDev,minSamples=minSamples)
    labelsPred = db.labels_

    unique_labels = set(labelsPred)
    core_samples_mask = np.zeros_like(labelsPred, dtype=bool) # create a corresponding array containing bool, acts like a map of the values
    core_samples_mask[db.core_sample_indices_] = True # set to True the coordinates of the not-in-core-values (i.e. values that are not in areas registered as high-density areas)

    colors = [plt.cm.rainbow(each) for each in np.linspace(0, 1, len(unique_labels))]

    if plot:
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labelsPred == k

            markersCore = markersList[class_member_mask & core_samples_mask] # get the list of markers for core values
            markersNotCore = markersList[class_member_mask & ~core_samples_mask] # same for not-core values

            xyCore = points[class_member_mask & core_samples_mask] # get the list of points in core
            xyNotCore = points[class_member_mask & ~core_samples_mask] # same for the not core points

            for j in range(len(markersCore)):
                plt.plot(
                    xyCore[j, 0],
                    xyCore[j, 1],
                    markersCore[j],
                    markerfacecolor=tuple(col),
                    markeredgecolor=tuple(col),
                    markersize=14,
                )

            for j in range(len(markersNotCore)):
                plt.plot(
                    xyNotCore[j, 0],
                    xyNotCore[j, 1],
                    markersNotCore[j],
                    markerfacecolor=tuple(col),
                    markeredgecolor=tuple(col),
                    markersize=6,
                )

    try :
        silhouetteScore = round(metrics.silhouette_score(points,labelsPred),3)
    except :
        silhouetteScore = -1

    randScore = 0

    for i in range(len(allLabel)):
        score = round(metrics.adjusted_rand_score(allLabel[i],labelsPred),3)
        if score > randScore:
            randScore = score

    if plot:
        plt.title(f"Estimated number of clusters: {n_clusters}\nSilhouette score = {silhouetteScore} ; Max Rand score = {randScore}\n Standard deviation: {maxStdDev} wt. %")

        legendElements = [plt.Line2D([],[],marker=".",markerfacecolor="k",markersize=10,mec = "k",label="Voreppe\n(PIS, PRO, REY, POM)",ls=""),plt.Line2D([],[],marker="x",markerfacecolor="k",markersize=7,mec = "k",label="Chenaillet (CHE)",ls=""),plt.Line2D([],[],marker="+",markerfacecolor="k",markersize=8,mec = "k",label="Corse (COR)",ls=""),plt.Line2D([],[],marker="1",markerfacecolor="k",markersize=8,mec = "k",label="Forville (FOR)",ls="")]#,plt.Line2D([],[],marker="3",markerfacecolor="k",markersize=8,mec = "k",label="Chenaillet supp. (Philippe)",ls="")]
        plt.legend(handles=legendElements)
        plt.xlabel(nameXAxis+" (wt. %)")
        plt.ylabel(nameYAxis+" (wt. %)")
        if scaled:
            minValues, maxValues = min(X+Y), max(X+Y)
            plt.axis("scaled")
            plt.ylim(minValues,maxValues)
            plt.xlim(minValues,maxValues)
            plt.margins(0.1)

        if saveFig:
            plt.savefig("".join(xNumEle)+"_"+"".join(yNumEle)+f"_RS-{randScore}.pdf")

        plt.show()

    else :
        return nameXAxis, nameYAxis, randScore, silhouetteScore, maxStdDev, (xNumEle,yNumEle)

def getAllLabelPossib():
    global allLabelPossib
    allLabelPossib = varPossibCombi([0,1,2,3],4)

## Test all element combinaisons

def possibCombi(elements,lenght):
    """Return a list with all the possible combinaisons that can be made with all elements given and limited in size by the lenght given"""

    if lenght==1:
        return [[ele] for ele in elements]

    poss = possibCombi(elements,lenght-1)
    L=[ele for ele in poss]

    for i in range(len(poss)):
        for j in range(len(elements)):
            data = poss[i]+[elements[j]]
            if elements[j] not in poss[i] and data not in L:
                L.append(data)

    return L

def varPossibCombi(elements,lenght):
    """Return a list with all the possible combinaisons that can be made with all elements given and limited in size by the lenght given"""

    if lenght==1:
        return [[ele] for ele in elements]

    poss = varPossibCombi(elements,lenght-1)
    L=[]

    for i in range(len(poss)):
        for j in range(len(elements)):
            data = poss[i]+[elements[j]]
            L.append(data)

    return L


def deleteDouble(L):
    """Delete doubles that will be interpreted the same way
    ex : ["Si","Al"] and ["Al","Si"] are the same"""

    possSets = [set(ele) for ele in L]
    treatedPoss = []
    #indexSameSet = [-1]*len(possSets)
    for i in range(len(possSets)):
        #if possSets[i] in possSets[:i]+possSets[i+1:] and possSets[i] not in treatedPoss:
        if possSets[i] not in treatedPoss:
            #indexSameSet[i] = 1
            treatedPoss.append(possSets[i])

    return [list(ele) for ele in treatedPoss]


def testAll(POSS,maxLenght,*,specifiedElements=[], epsilon=1):
    """Test all the plot possibilities"""

    getAllLabelPossib() # computes the allLabelCombi global variable

    allUniquePoss = POSS

    table = open(f"cluster_{maxLenght}_{getDate()}.csv","w")
    table.write("X,Y,maxRandScore,silhouetteScore,stdDev\n")

    startTime = dt.now()

    for i in range(1,len(allUniquePoss)):
        for j in range(i):
            nameXAxis, nameYAxis, randScore, silhouetteScore,maxStdDev, elmts = cluster(allUniquePoss[i],allUniquePoss[j],plot=False, epsilon=epsilon)
            #print(nameXAxis, nameYAxis, randScore)
            if randScore >= 0.6 and silhouetteScore >= 0.6:
                table.write(f"{nameXAxis},{nameYAxis},{randScore},{silhouetteScore},{maxStdDev},{elmts}\n")

    table.close()

    stopTime = dt.now()
    print(f"Temps d'execution : {stopTime-startTime}")



def getDate():
    NOW = dt.now()
    return f"{NOW.year}-{NOW.month}-{NOW.day}_{NOW.hour}-{NOW.minute}-{NOW.second}"


def getBestCombi(fileName):
    """Get the combinaisons in the cluster results file that have the best (and second best randScore) randScore and within them, the best silhouette score"""
    file = open(f"{fileName}")
    SCORES = np.loadtxt(f"{fileName}",skiprows=1,usecols=(2,3),delimiter=",")

    ELMT = np.loadtxt(f"{fileName}",dtype=str,skiprows=1,usecols=(0,1),delimiter=",")
    n,p = np.shape(SCORES)

    maxRandScore = max(SCORES[:,0])
    maxRandScoreLabel = SCORES[:,0] == maxRandScore
    maxSilhouetteScore = max(SCORES[:,1][maxRandScoreLabel])
    maxSilhouetteScoreLabel = SCORES[:,1] == maxSilhouetteScore


    secondMaxRandScore = max(SCORES[:,0][~maxRandScoreLabel])
    secondMaxRandScoreLabel = SCORES[:,0] == secondMaxRandScore
    secondMaxSilhouetteScore = max(SCORES[:,1][secondMaxRandScoreLabel])
    secondMaxSilhouetteScoreLabel = SCORES[:,1] == secondMaxSilhouetteScore

    return maxRandScore, ELMT[:,0:2][maxRandScoreLabel],secondMaxRandScore, ELMT[:,0:2][secondMaxRandScoreLabel],"###",maxSilhouetteScore,ELMT[:,0:2][maxRandScoreLabel & maxSilhouetteScoreLabel],secondMaxSilhouetteScore,ELMT[:,0:2][secondMaxRandScoreLabel & secondMaxSilhouetteScoreLabel]

def tablBestCombi(fileName):
    file = open(f"bestCombi_{fileName[:-4]}.csv","w")

    DATA = getBestCombi(fileName)

    maxRScore = DATA[0]
    maxSScore = DATA[5]
    maxElmt = DATA[6]

    secondRScore = DATA[2]
    secondSScore = DATA[7]
    secondElmt = DATA[8]

    file.write("Abscisse,Ordonnée,RS,SS\n")

    for i in range(len(maxElmt)):
        file.write(maxElmt[i][0] + "," + maxElmt[i][1] + f",{maxRScore},{maxSScore}\n" )
    for i in range(len(secondElmt)):
        file.write(secondElmt[i][0] + "," + secondElmt[i][1] + f",{secondRScore},{secondSScore}\n" )

    file.close()

#
#
#

#elementOK = ["Si","P","K","Ca","Ti","Fe","Zn","Rb","Sr","Zr"]
#elementPOK = ["Na","Al","V","Cr","Cu"]

print("For the first run or when shell has been reinistilised, do not forget to run >getAllLabelPossib()< at first.\n")
