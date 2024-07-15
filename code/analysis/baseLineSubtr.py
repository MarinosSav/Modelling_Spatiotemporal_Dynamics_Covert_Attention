recHz = eyelinkData["SETTINGS"]["SAMPLING_RATE"]
print(base_filename + " recorded at: " + str(recHz) + "Hz")

pBasePerHz = np.array(np.multiply(pBasePer, recHz / 1000), dtype=int)
pERPROriAmpPerHz = np.array(np.multiply(pERPROriAmpPer, recHz / 1000), dtype=int)
pERPROriAreaPerHz = np.array(np.multiply(pERPROriAreaPer, recHz / 1000), dtype=int)
pPostERPRPerHz = int(pPostERPRPer * recHz / 1000)

respBase = np.nanmean(erprMatrix[:, pBasePerHz[0] : pBasePerHz[1]], 1)
respBaseGaze = []
respBaseGaze.append(np.nanmean(gazeMatrix[:, pBasePerHz[0] : pBasePerHz[1],0], 1)) 
respBaseGaze.append(np.nanmean(gazeMatrix[:, pBasePerHz[0] : pBasePerHz[1],1], 1))

erprMatrixBaseSubt = (
    erprMatrix
    - np.tile(
        respBase,
        (np.shape(erprMatrix)[1], 1),
    ).T
)
gazeMatrixBaseSubt = gazeMatrix.copy()
for g in range(np.shape(gazeMatrixBaseSubt)[2]):
    gazeMatrixBaseSubt[:,:,g] = (
        gazeMatrix[:,:,g]
        - np.tile(
            respBaseGaze[g],
            (np.shape(gazeMatrix)[1], 1),
        ).T
    )
# gazeMatrixNorm = gazeMatrix.copy()
# for g in range(np.shape(gazeMatrixNorm)[2]):
#     gazeMatrixNorm[:,:,g] = gazeMatrix[:,:,g]-np.mean(gazeMatrix[:,:,g])
gazeMatrixPolar = gazeMatrixBaseSubt.copy()
gazeMatrixPolar[:,:,0],gazeMatrixPolar[:,:,1] = cart2pol(gazeMatrixBaseSubt[:,:,0], gazeMatrixBaseSubt[:,:,1])
