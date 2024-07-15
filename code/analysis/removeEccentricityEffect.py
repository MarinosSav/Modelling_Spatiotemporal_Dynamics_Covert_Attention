rho,phi = cart2pol(probeLocationCoor[:,0],probeLocationCoor[:,1])

# from scipy.stats import pearsonr 
# pearsonr(rho,mainVarPupilData)
if showPlotsPerObs:
    fitColors = ['r','b']
    plt.figure()
    plt.plot(rho,mainVarPupilData,'ko')

for lf in range(2):
    fitParams = np.polyfit(rho,mainVarPupilData,lf+1)
    xFitData = np.linspace(min(rho),max(rho),10)
    fitVals = np.polyval(fitParams,xFitData) 
    
    if showPlotsPerObs:
        plt.plot(xFitData,fitVals,fitColors[lf] + 'o-')

mainVarPupilData = mainVarPupilData-np.polyval(fitParams,rho) 
