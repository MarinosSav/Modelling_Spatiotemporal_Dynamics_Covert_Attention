
curPsychopyFileName = ""
psychopyFileNames = searchForFileInSubDir(ppName + "*.csv", psychPyFilesDir)

for psychopyFileName in psychopyFileNames:
    if ppName + "_" + "sess00" + str(sessNum) in psychopyFileName:
        curPsychopyFileName = psychopyFileName
        break
if curPsychopyFileName == "":
    print("ERROR: csv file of psychopy not found for: ")
    print(ppName + "_sess00" + str(sessNum))
else:
    print("Loading: " + curPsychopyFileName)

df = pd.read_csv(curPsychopyFileName)
if df["key_skip_tutorial.keys"][1] == "q":  # tutorial skipped
    firstInd = np.where(np.isfinite(df["time_cue_probe"]))[0][0]
else:
    # firstPractBlockInd = np.where(np.isfinite(df["time_cue_probe"]))[0][0]
    # firstInd = (
    #     np.where(~np.isfinite(df["time_cue_probe"][firstPractBlockInd:]))[0][0]
    #     + firstPractBlockInd
    #     + 1
    # )
    firstInd = np.where(np.isfinite(df["time_cue_probe"]))[0][0]
    # firstInd = (
    #     np.where(~np.isfinite(df["time_cue_probe"][firstPractBlockInd:]))[0][0]
    #     + firstPractBlockInd
    #     + 1
    # )
lastInd = np.where(~np.isfinite(df["time_cue_probe"][firstInd:]))[0][0] + firstInd

targetID = np.array(df["target_stim"][firstInd:lastInd])
targetID[targetID == "E"] = np.char.lower(targetID[targetID == "E"].astype(str)).astype(object)

responseID = np.array(df["key_resp.keys"][firstInd:lastInd])
cueProbeSOA = np.array(df["time_cue_probe"][firstInd:lastInd])
cueTargetSOA = np.array(df["time_cue_target"][firstInd:lastInd])
probeLocation = np.array(df["probe_location"][firstInd:lastInd])
cueLocation = np.array(df["cue_start_location"][firstInd:lastInd])
targetDirection = np.array(df["next_direction"][firstInd:lastInd])