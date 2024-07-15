# rho,phi = cart2pol(probeLocationCoor[:,0],probeLocationCoor[:,1])

# LEFT RIGHT
if showPlotsPerObs:
    fitColors = ['r','b']
    plt.figure()
    plt.plot(probeLocationCoor[:,0],mainVarPupilData,'ko')
    plt.title('LEFT-RIGHT')

for lf in range(2):
    fitParams = np.polyfit(probeLocationCoor[:,0],mainVarPupilData,lf+1)
    xFitData = np.linspace(min(probeLocationCoor[:,0]),max(probeLocationCoor[:,0]),10)
    fitVals = np.polyval(fitParams,xFitData) 
    
    if showPlotsPerObs:
        plt.plot(xFitData,fitVals,fitColors[lf] + 'o-')

mainVarPupilData = mainVarPupilData-np.polyval(fitParams,rho) 

# TOP DOWN
if showPlotsPerObs:
    fitColors = ['r','b']
    plt.figure()
    plt.plot(probeLocationCoor[:,1],mainVarPupilData,'ko')
    plt.title('TOP DOWN')
    
for lf in range(2):
    fitParams = np.polyfit(probeLocationCoor[:,1],mainVarPupilData,lf+1)
    xFitData = np.linspace(min(probeLocationCoor[:,1]),max(probeLocationCoor[:,1]),10)
    fitVals = np.polyval(fitParams,xFitData) 
    
    if showPlotsPerObs:
        plt.plot(xFitData,fitVals,fitColors[lf] + 'o-')

mainVarPupilData = mainVarPupilData-np.polyval(fitParams,rho) 
