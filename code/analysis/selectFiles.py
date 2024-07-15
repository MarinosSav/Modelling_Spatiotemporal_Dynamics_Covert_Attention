from classes.filename_functions import (
    searchForFileInSubDir,
    fileSelectionPopUp,
    filesSelectionPopUp,
    dirSelectionPopUp,
    splitPathFilenameComponents,
)


def selectFiles(fileSelectOption, dirPath=None, fileExts=["*.avi", "*.mp4", "*.MOV", "*.webm"]):
    # Find video files in folder
    if fileSelectOption == "single_filename":
        logger.log(
            "Select one file name with extensions: " + ",".join(fileExts),
            logType="info",
        )
        fileNamesWithPath = [fileSelectionPopUp()]
        dirPath, __, __, __, file_extension = splitPathFilenameComponents(fileNamesWithPath)
    elif fileSelectOption == "directory":

        if dirPath == None:
            logger.log(
                "Select directory containing files with extensions: " + ",".join(fileExts),
                logType="info",
            )
            dirPath = dirSelectionPopUp()

        # THIS CODE IS IMPROVED IN Science/Codes/Irissometry/Python!!!
        # that new code returns dirPaths and extensions per videofile
        # copy paste to current code if this feature is needed
        if len(fileExts) == 1:
            file_extension = fileExts[0][1:]
        else:
            file_extension = []
        fileNamesWithPath = searchForFileInSubDir(fileExts, dirPath)
    elif fileSelectOption == "multiple_filenames":
        logger.log(
            "Select multiple file names with extensions: " + ",".join(fileExts),
            logType="info",
        )
        fileNamesWithPath = filesSelectionPopUp()

        dirPath, __, __, __, file_extension = splitPathFilenameComponents(fileNamesWithPath[0])
    else:
        fileNamesWithPath = []
        dirPath = []
        logger.log(
            "File selection option indicated in settings unknown: " + fileSelectOption,
            logType="error",
        )

    # TO DO: check extension/codec and warn if WEBM/VP8/VP9 because of wrong FPS, timestamps, etc.

    return fileNamesWithPath, dirPath, file_extension
