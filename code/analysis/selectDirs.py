# from classes.filename_functions import (
#     dirsSelectionPopUp,
# )

import tkfilebrowser
import tkinter as tk


def selectDirs(initialDir="", fileExts=[".edf"]):
    print("dir")
    print(initialDir)
    # root = tk.Tk()
    logger.log(
        "Select one or multiple directories containing files with extensions: "
        + ",".join(fileExts),
        logType="info",
    )
    logger.log(
        "(!!!Popup window might be hidden under another window!!!)",
        logType="info",
    )
    dirNamesWithPath = tkfilebrowser.askopendirnames(
        initialdir=initialDir,
        title="Choose one or multiple directories with " + ",".join(fileExts) + " files",
    )
    return dirNamesWithPath
