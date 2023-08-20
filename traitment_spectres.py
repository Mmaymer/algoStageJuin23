
# Martin AYMÉ
# E.N.S. de Lyon
# Python, UTF-8
# June-July 2023

## Bibliothèques

import os
import numpy as np
import matplotlib.pyplot as plt


## Formatting

def main(directory):
    """Format the .mca files so that they can be red by CloudCal for the calibrations"""
    allFiles = os.listdir(directory)
    try:
        os.mkdir(f"format_{directory}_sensorNbr")
    except:
        if int(input(f"Directory has already been formated (see <format_{directory}_sensorNbr>).\nDo you want to format it again ? (previous data will be erased)\n0 : No\t1 : Yes\n> ")) == 1:
            eraseDir(directory)
        else:
            return

    allMca = [file for file in allFiles if file[-3:] == "mca"] # select only .mca
    # files

    for file in allMca: # format all files in form_in dorectory
        if file[:7] == "Test_#_":
            newName = createNewName(file)
            os.rename(f"{directory}/{file}",f"{directory}/{newName}")
            file = newName
        formattingSensorNbr(file,directory)

    print(f"*#* ALL FILES IN <{directory}> FORMATED IN <format_{directory}_sensorNbr> *#*")


def eraseDir(directory):
    for file in os.listdir(f"format_{directory}"):
        os.remove(f"format_{directory}//{file}")


def createNewName(file):
    return file[:4] + file[7:]


def formattingSensorNbr(file,directory):

    f = open(f"{directory}/{file}") # open the selected file in form_in dir.
    content = f.readlines() # .cma file content recuperation
    fileName = file.split("_") # get info stocked in file name

    voltage = round(float(content[18][:-1].split("=")[-1])) #keV
    scaleSlope = round(float(content[14][:-1].split("=")[-1]),1)/1000 #keV
    offset = round(float(content[15][:-1].split("=")[-1]),1)/1000 #keV
    sample = fileName[0]
    number = "#"+fileName[1]
    beam = fileName[2][4]
    date = fileName[3]+"-"+fileName[4]+"-"+fileName[5]
    time = fileName[6]+"-"+fileName[7]+"-"+fileName[8]+"_"+fileName[9][:-5]

    name = f"{sample}_{number}_B{beam}-{voltage}keV_{date}_{time}" # name of the spectrum

    NewF = open(f"format_{directory}_sensorNbr\{name}.csv","w") # create new .csv file

    noSpaceStriped = [content[i][:-1].replace(" ","") for i in range(21)]
    INFO = [line.split("=") for line in noSpaceStriped]
    # switch info (example : "Tube keV = 40.062" to "Tube keV,40.062") to write
    # them in the new .csv file

    infoModel = ['Duration Time', 'Ambient Temperature', 'Detector Temperature', 'Valid Accumulated Counts', 'Raw Accumulated Counts', 'Valid Count Last Packet', 'Raw Count Last Packet', 'Live Time', 'HV DAC', 'HV ADC', 'Filament DAC', 'Filament ADC', 'Pulse Length', 'Pulse Period', 'Filter', 'eV per channel', 'Number of Channels','Vacuum']

    NewF.write(f"Spectrum_{name}\n") # name of the file
    NewF.write(f"{fileName[0]}\n") # nb of the test

    counts = [int(content[i+23].strip()) for i in range(2048)]
    totalCounts = sum(counts) # total number of counts
    infoOrder = [17,-1,-1,-2,-1,-1,-1,16,-1,18,-1,19,-1,-1,20,14,5,-1] # info order from Label

    for i in range(len(infoModel)):
        NewF.write(f"{infoModel[i]}") # copying the raw info
        if infoOrder[i] != -1: # if specific info is available in the mca file
            NewF.write(f",{INFO[infoOrder[i]][1]}\n")
        elif infoOrder[i] == -2: # case of total counts
            NewF.write(f",{totalCounts}\n")
        else :
            NewF.write(f"\n") # else, nothing is written

    NewF.write("Channel#,Intensity\n")

    for j in range(0,2048):
        # energy = round(j*scaleSlope+offset,6) # energy calc. for each channel
        # gap of <scaleSlope> keV between each channel
        # offset of <offset> keV
        NewF.write(f"{j},{int(content[23+j][:-1])}\n")

    NewF.close() # save and close the file



def mainEnergy(directory):
    """Format .mca files so that they can be red by Spectragryph"""
    allFiles = os.listdir(directory)
    try:
        os.mkdir(f"format_{directory}_energy")
    except:
        if int(input(f"Directory has already been formated (see <format_{directory}>).\nDo you want to format it again ? (previous data will be erased)\n0 : No\t1 : Yes\n> ")) == 1:
            eraseDir(directory+"_energy")
        else:
            return

    allMca = [file for file in allFiles if file[-3:] == "mca"] # select only .mca
    # files

    for file in allMca: # format all files in form_in dorectory
        if file[:7] == "Test_#_":
            newName = createNewName(file)
            os.rename(f"{directory}/{file}",f"{directory}/{newName}")
            file = newName
        formattingEnergy(file,directory)

    print(f"*#* ALL FILES IN <{directory}> ARE FORMATED IN format_{directory}_energy *#*")

def formattingEnergy(file,directory):

    f = open(f"{directory}/{file}") # open the selected file in form_in dir.
    content = f.readlines() # .cma file content recuperation
    fileName = file.split("_") # get info stocked in file name

    voltage = round(float(content[18][:-1].split("=")[-1])) #keV
    scaleSlope = round(float(content[14][:-1].split("=")[-1]),1)/1000 #keV
    offset = round(float(content[15][:-1].split("=")[-1]),1)/1000 #keV
    sample = fileName[0]
    number = "#"+fileName[1]
    beam = fileName[2][4]
    date = fileName[3]+"-"+fileName[4]+"-"+fileName[5]
    time = fileName[6]+"-"+fileName[7]+"-"+fileName[8]+"_"+fileName[9][:-5]

    name = f"{sample}_{number}_B{beam}-{voltage}keV_{date}_{time}" # name of the spectrum

    NewF = open(f"format_{directory}_energy\{name}.csv","w") # create new .csv file

    noSpaceStriped = [content[i][:-1].replace(" ","") for i in range(21)]
    INFO = [line.split("=") for line in noSpaceStriped]
    # switch info (example : "Tube keV = 40.062" to "Tube keV,40.062") to write
    # them in the new .csv file

    infoModel = ['Duration Time', 'Ambient Temperature', 'Detector Temperature', 'Valid Accumulated Counts', 'Raw Accumulated Counts', 'Valid Count Last Packet', 'Raw Count Last Packet', 'Live Time', 'HV DAC', 'HV ADC', 'Filament DAC', 'Filament ADC', 'Pulse Length', 'Pulse Period', 'Filter', 'eV per channel', 'Number of Channels','Vacuum']

    NewF.write(f"Spectrum_{name}\n") # name of the file
    NewF.write(f"{fileName[0]}\n") # nb of the test

    counts = [int(content[i+23].strip()) for i in range(2048)]
    totalCounts = sum(counts) # total number of counts
    infoOrder = [17,-1,-1,-2,-1,-1,-1,16,-1,18,-1,19,-1,-1,20,14,5,-1] # info order from Label

    for i in range(len(infoModel)):
        NewF.write(f"{infoModel[i]}") # copying the raw info
        if infoOrder[i] != -1: # if specific info is available in the mca file
            NewF.write(f",{INFO[infoOrder[i]][1]}\n")
        elif infoOrder[i] == -2: # case of total counts
            NewF.write(f",{totalCounts}\n")
        else :
            NewF.write(f"\n") # else, nothing is written

    NewF.write("Energy (keV),Intensity (cps)\n")

    for j in range(0,2048):
        energy = round(j*scaleSlope+offset,6) # energy calc. for each channel
        # gap of <scaleSlope> keV between each channel
        # offset of <offset> keV
        NewF.write(f"{energy},{int(content[23+j][:-1])}\n")

    NewF.close() # save and close the file


def sort(directory):
    """Sort the formated files in different directories according to the voltage of the test"""
    allDirect = os.listdir(directory)
    allFiles = [f for f in allDirect if os.path.isfile(f"{directory}/{f}")] # extract only the files

    voltages = {}

    for i in range(len(allFiles)):
        strFileVoltage = allFiles[i].split("_")[2][3:5]
        num = [str(i) for i in range(10)]
        strNumFileVoltage = [strFileVoltage[i] for i in range(len(strFileVoltage)) if strFileVoltage[i] in num]
        fileVolt = round(float("".join(strNumFileVoltage))) # extraction of the voltage value

        if fileVolt not in voltages:
            voltages.update({fileVolt:[i]})
        else :
            voltages[fileVolt].append(i)

    listVoltages = list(voltages.keys())
    for i in range(len(listVoltages)):
        if f"{listVoltages[i]}keV" in os.listdir(directory):
            if input(f"Directory <{directory}/{listVoltages[i]}keV> already exists. Do you want to make another or use the existing one (existing data may be erased if sameName issue is raised) ?\n0 : Create new directory (erase the existing one)\t1 : Continue withe existing directory") == 1:
                print("Sorting...")
            else :
               [os.remove(f"{directory}/{file}") for file in os.listdir(directory)]
        else :
            os.mkdir(f"{directory}/{listVoltages[i]}keV") # create directory
        for index in voltages[listVoltages[i]]:
            os.replace(f"{directory}/{allFiles[index]}",f"{directory}/{listVoltages[i]}keV/{allFiles[index]}") # move file


## Filters

def filtre(directory):
    allFiles = os.listdir(directory)
    n=len(allFiles)
    U,I,F = [0]*n,[0]*n,[0]*n

    for i in range(n):
        U[i],I[i],F[i] = getInfo(allFiles[i],directory)

    P = [U[i]*I[i] for i in range(len(U))]
    plt.plot(F,U,label="U=f(F)")
    plt.plot(F,I,label="I=f(F)")
    plt.plot(F,P,label="P=f(F)")
    plt.legend()
    plt.show()



def getInfo(file,directory):
    f = open(f"{directory}/{file}")
    content = f.readlines()

    U = round(float(content[18][:-1].split("=")[-1]),2) #keV
    I = round(float(content[19][:-1].split("=")[-1]),2) #µA
    F = int(content[20][:-1].split("=")[-1]) #n°

    return U,I,F


## Coups/I

#cps=[0.7,0.89,1.01,1.25,1.42,1.56]
#i=[75,100,125,150,175,200]
#plt.plot(cps,i)
#plt.show()

## Lines smoothening

def smoothen(directory,multiplicity):
    """To smoothen the signal of a sample, several spectra are recorded with the pXRF and then, for each sensor (i.e. for each energy gap) we calculate the mean between all spectra, the result is then saved as a "averaged" spectrum

    multiplicity (list or int) : can be aither an integer or a list. If integer, represents the lenght of groups that are formed and averaged. If list, contains the length of ALL THE GROUPS that are to be created (ex : [2,3,5] will created 3 groups of 2, 3 and 5 spectra -- so it requires exactly 10 spectra)."""

    allFiles = os.listdir(directory)
    try:
        os.mkdir(f"{directory}/{directory}_smoothened")
    except:
        if int(input(f"Directory has already been smoothened (see <{directory}/{directory}_smoothened>).\nDo you want to smoothen it again ? (previous data may be erased)\n0 : No\t1 : Yes\n> ")) == 1:
            #eraseDirSmooth(directory)
            print("Formatting...")
        else:
            return

    allCsv = [file for file in allFiles if file[-3:] == "csv"] # select only .csv
    # files
    if  str(type(multiplicity))=="<class 'int'>":
        allCsvGrouped = [[allCsv[j] for j in range(i,i+multiplicity)] for i in range(0,len(allCsv),multiplicity)]
        # the multiplicity is the number of spectra recorded for each sample

    else : # if the multiplicity is a list, then groups are formed according to it
        cumulCrescIndex = multiplicityCumul(multiplicity)
        allCsvGrouped = [[allCsv[j] for j in range(cumulCrescIndex[0])]]
        allCsvGrouped += [[allCsv[j] for j in range(cumulCrescIndex[i-1],cumulCrescIndex[i])] for i in range(1,len(multiplicity))]

    for group in allCsvGrouped: # smoothen all files in directory
        for file in group:
            if file[:7] == "Test_#_":
                newName = createNewName(file)
                os.rename(f"{directory}/{file}",f"{directory}/{newName}")
                file = newName
        spectralMean(group,directory)

    print(f"*#* ALL FILES IN <{directory}> SMOOTHENED IN <{directory}/{directory}_smoothened> *#*")
    return

def spectralMean(group,directory):
    """Average the n spectra in a unique averaged spectrum, with n the group lenght"""

    spectraNumbers = ""
    allContents = []
    files = []

    for i in range(len(group)): # for each file, extraction of the spectrum numbre and data
        number = group[i].split("_")[1]

        if i == len(group)-1:
            spectraNumbers += number
        else :
            spectraNumbers += number+"-"

        baseFile = open(f"{directory}/{group[i]}") # open the i-th file

        allContents.append(baseFile.readlines()) # extract the data

        baseFile.close()


    nameAfterFilename = ["",""] # extract the name structure from the non-averaged spectra
    # before and after the spectrum number

    nameAfterFilename[0] = group[0].split(spectraNumbers.split("-")[0])[0]
    nameAfterFilename[1] = group[0].split(spectraNumbers.split("-")[0])[1]

    name = nameAfterFilename[0] + spectraNumbers + nameAfterFilename[1]

    for i in range(len(allContents)):
        allContents[i] = [allContents[i][j].split(",") for j in range(len(allContents[i]))]
    searchForStart = [allContents[0][i][0] for i in range(len(allContents[0]))]

    start = searchForStart.index("Vacuum\n")

    averageFile = open(f"{directory}/{directory}_smoothened/{name}","w")

    averageFile.write(name+"\n")

    for i in range(1,start+2):
        try :
            averageFile.write(f"{allContents[0][i][0]},{allContents[0][i][1]}")
        except :
            averageFile.write(f"{allContents[0][i][0].strip()},\n")

    # if allContents[0][start+2][0] == int(allContents[0][start+2][0]):
    #     averageFile.write("Channel#,Intensity_(cps)\n")
    # else :
    #     averageFile.write("Energy_(keV),Intensity_(cps)\n")

    for i in range(start+2,len(allContents[0])):
        average = 0
        for j in range(len(group)):
            #print(allContents[j][i][0],allContents[j][i][1])
            average += int(allContents[j][i][1])

        averageFile.write(f"{float(allContents[j][i][0])},{average//len(group)}\n")

    averageFile.close()


def eraseDirSmooth(directory):
    for file in os.listdir(f"{directory}/{directory}_smoothened"):
        os.remove(f"{directory}/{directory}_smoothened//{file}")

def multiplicityCumul(multip):
    """From a list of groups length, return a list of cumulated crescent index"""
    multiCumul = [0]*len(multip)
    N = 0
    for i in range(len(multip)):
        multiCumul[i] = N + multip[i]
        N += multip[i]
    return multiCumul


## Start message

print("Here are some functions to format your spectra :\n")
print(">> main(directory) : format the spectra (.mca) in \"directory\" so that they can be used in CloudCal\n")
print(">> mainEnergy(directory) : format the spectra (.mca) in \"directory\" so that they can be used in Spectragryph with the right abscisse\n")
print(">> sort(directory) : sort the \"directory\" according to the various beams\n")
print(">> smoothen(directory,multiplicity) : make the average for each energy canal with a \"multiplicity\" step")
