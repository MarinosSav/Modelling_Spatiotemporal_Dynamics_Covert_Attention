nXBins = np.shape(allErprAmp2DMapPerTimeBin_ave_ref)[1]
nYBins = np.shape(allErprAmp2DMapPerTimeBin_ave_ref)[0]


varCBarRange = [
    np.percentile(allErprAmp2DMapPerTimeBin_ave_ref[:], 1),
    np.percentile(allErprAmp2DMapPerTimeBin_ave_ref[:], 99),
]

nTimeBins = np.shape(allErprAmp2DMapPerTimeBin_ave_ref)[2]


nFrames = 100


cm = (plt.cm.coolwarm_r(np.linspace(0, 1, 256)) * 255).astype("uint8")[:, 0:3]


def customCMAP(im_grey, cmap):
    im_color = np.zeros((np.shape(im_grey)[0], np.shape(im_grey)[1], 3))
    for y in range(np.shape(im_grey)[0]):
        for x in range(np.shape(im_grey)[1]):
            for c in range(3):
                im_color[y, x, c] = cmap[im_grey[y, x], c]
    im_color = im_color.astype("uint8")
    return im_color


allTempMap = np.zeros((nYBins, nXBins, nFrames))
allTempMap[:] = np.nan

y = np.linspace(0, nYBins - 1, nYBins).astype("int")
x = np.linspace(0, nXBins - 1, nXBins).astype("int")
z = np.linspace(0, nTimeBins - 1, nTimeBins).astype("int")

# yi = np.linspace(0, nYBins-1, nYBins)
# xi = np.linspace(0, nXBins-1, nXBins)
# zi = np.linspace(0, nTimeBins-1, nFrames)

# Vi = interpn((y, x, z), allErprAmp2DMapPerTimeBin_ave_ref, np.array([xi, yi, zi]).T, method='linear', bounds_error=False, fill_value=0)

from scipy.interpolate import RegularGridInterpolator

fn = RegularGridInterpolator((y, x, z), allErprAmp2DMapPerTimeBin_ave_ref, method="linear")

for xi in x:
    for yi in y:
        countZi = 0
        for zi in np.linspace(0, nTimeBins - 1, nFrames):
            pts = fn(np.array([[yi, xi, zi]]))
            allTempMap[yi, xi, countZi] = pts
            countZi += 1


out = cv2.VideoWriter(
    "Figures/averageAcrossObservers.avi",
    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    25,
    (1600, 800),
)
countBT = -1
for bT in range(nFrames):
    countBT = countBT + 1
    tempMap = allTempMap[:, :, countBT]
    curCBarRange = [np.min(tempMap[:]), np.max(tempMap[:])]
    tempMap = tempMap - np.min(tempMap[:])
    tempMap = tempMap / np.max(tempMap[:])
    tempMap = tempMap * (np.diff(curCBarRange)[0] / np.diff(varCBarRange)[0])
    tempMap = tempMap + (curCBarRange[0] - varCBarRange[0]) / np.diff(varCBarRange)[0]

    # tempMap = tempMap - np.min(allTempMap[:])
    # tempMap = tempMap / np.max(allTempMap[:])
    tempMap[tempMap[:] < 0] = 0
    tempMap[tempMap[:] > 1] = 1
    tempMap = np.uint8(tempMap * 255)

    im_color = customCMAP(np.flipud(tempMap), cm)

    im_color = cv2.resize(im_color, (1600, 800), interpolation=cv2.INTER_CUBIC)

    # cv2.imshow('Frame',im_color)

    out.write(im_color)
    # cv2.waitKey(0)

out.release()
# cv2.destroyAllWindows()
