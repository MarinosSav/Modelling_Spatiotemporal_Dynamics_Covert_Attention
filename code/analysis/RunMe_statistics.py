# TO DO: ask Sam to do cluster permutation test, also per target location (output to matlab; implement python-based cluster permutation test)
# TO DO: Fit both a linear fit with a certain slope TOGETHER with normal distribution/mexican hat!



# TO DO: calculate accuracies per session, and per target location, and per move direction
# TO DO: collapse data across sessions per subject --> do two-way anova on slopes with factor session num


normTypeName = "z"  # "z" or "None"
varOfInterestAbr = "amp"  # "amp" or "area"
# varOfInterestAbr = "area"  # "amp" or "area"
onsetEvent = "cue"  # "cue" or "probe"
pupilUnit = "z"  # "z" or "au" (None)

# Effect of eccentricity is removed by subtracting average 2D Map per observer or pooled across observers
refMap = "Obs"  # "Obs" or "Pool" or "None"

if varOfInterestAbr == "area":
    colormapper = "coolwarm_r"
elif varOfInterestAbr == "amp":
    colormapper = "coolwarm"

exec(open("import_packages.py").read())

pickleFileName = "data/allVars.pickle"
with open(pickleFileName, "rb") as handle:
    allPPName, allSessNum, allNOutliers, allAccuracy, allPercFewSamplesTrials, allPercBadFixTrials = pickle.load(handle)
    handle.close()

exec(open("concatenateERPR_perXYTimeBin.py").read())
exec(open("concatenateERPR_perXYTimeBinPerTarLoc.py").read())
exec(open("concatenateERPR_perXTimeBin.py").read())
exec(open("concatenateERPR_perYTimeBin.py").read())
exec(open("concatenateERPR_perTimeBin.py").read())

exec(open("reference_subtraction.py").read())

exec(open("all_ERPR_perXYTimeBin.py").read())
exec(open("all_ERPR_perXYTimeBinPerTarLoc.py").read())
# exec(open("all_ERPR_perX_or_YTimeBin.py").read())
exec(open("all_ERPR_perTimeBin.py").read())

# exec(open("erpr_statistics.py").read())
exec(open("erpr_statistics_2D.py").read())
# exec(open("erpr_statistics_2D_highRes.py").read()) # REF SUBTRACTION NOT YET INCLUDED

exec(open("erpr_statistics_2D_perTarLoc.py").read())

exec(open("all_ERPR_timeLapse.py").read())

exec(open("all_ERPR_refMaps.py").read())