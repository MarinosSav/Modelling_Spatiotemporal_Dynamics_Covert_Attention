

# pupilERPRMin = np.nanmin(erprMatrix[:, pERPRPerHz[0] : pERPRPerHz[1]], 1)
# pupilERPRMax = np.nanmax(erprMatrix[:, pERPRPerHz[0] : pERPRPerHz[1]], 1)
pupilERPRStart = np.nanmax(erprMatrix[:, pBasePerHz[0] : pBasePerHz[1]], 1)
pupilERPREnd = np.nanmin(erprMatrix[:, pERPROriAmpPerHz[0] : pERPROriAmpPerHz[1]], 1)
pupilERPRAmp = pupilERPRStart - pupilERPREnd
pupilPostERPRMean = np.nanmean(erprMatrix[:, pPostERPRPerHz:], 1) - respBase


pupilERPRArea = np.divide(
    np.sum(erprMatrixBaseSubt[:, pERPROriAreaPerHz[0] : pERPROriAreaPerHz[1]], 1),
    pERPROriAreaPerHz[1] - pERPROriAreaPerHz[0],
)