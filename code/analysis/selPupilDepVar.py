# select dependent variable of interest
if varOfInterest == "ERPR Area":
    mainVarPupilData = pupilERPRArea.copy()
    varOfInterestAbr = "area"
elif varOfInterest == "ERPR Amplitude":
    mainVarPupilData = pupilERPRAmp.copy()
    varOfInterestAbr = "amp"
elif varOfInterest == "Post ERPR Amplitude":
    mainVarPupilData = pupilPostERPRMean.copy()
    varOfInterestAbr = "postAmp"
else:
    print("ERROR: STILL NEED TO define variable of interest: " + varOfInterest)

if varOfInterest == "ERPR Area":
    colormapper = "coolwarm_r"
elif varOfInterest == "ERPR Amplitude":
    colormapper = "coolwarm"