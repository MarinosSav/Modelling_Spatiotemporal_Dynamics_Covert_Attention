out = cv2.VideoWriter(
    "Figures/" + ppName + ".avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (1600, 800)
)

countBT = -1
for bT in range(np.max(nTimeBinsArr)):
    countBT = countBT + 1
    tempMap = erprAmp2DMapPerTimeBin_ave[:, :, countBT]
    tempMap = tempMap - np.min(tempMap[:])
    tempMap = tempMap / np.max(tempMap[:])
    tempMap = np.uint8(tempMap * 255)
    tempMap = cv2.resize(tempMap, (1600, 800))

    im_color = cv2.applyColorMap(tempMap, cv2.COLORMAP_TWILIGHT_SHIFTED)

    # cv2.imshow('Frame',im_color)

    out.write(im_color)
    # cv2.waitKey(0)

out.release()
# cv2.destroyAllWindows()
