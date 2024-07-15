from classes.eyetracker_parser import eyelink_parser

allERPRDict = {}

# select directories to search edf files and loop through each directory
dirNamesWithPath = selectDirs(initialDir)


logger.log("Parsing edf files by converting to ASCI and then pickle files.")
for dirName in dirNamesWithPath:
    # search edf files per directory
    edfFileNamesWithPath, __, __ = selectFiles("directory", dirPath=dirName, fileExts=["*.edf"])
    nEdfFiles = len(edfFileNamesWithPath)
    countSubplots = 0
    for edfFilename in edfFileNamesWithPath:

        from classes.eyetracker_parser import eyelink_parser

        eyelinkObj = eyelink_parser(edfFilename=edfFilename)
        eyelinkObj.suppressFileLoadMsg = True # DONT SET THIS TO TRUE STANDARD

        # parseObj.lookForMsg = ['trial_start']

        # convert edf to asc
        eyelinkObj.edf2asc()

        # parse ascii file and put eye-tracking data in dictionary
        eyelinkData = eyelinkObj.parseAscFile()

logger.log("Done parsing edf files.")
