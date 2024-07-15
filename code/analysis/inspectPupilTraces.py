#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot average pupil responses
xdata = np.linspace(0, 3, np.shape(erprMatrixBaseSubt)[1])
if showPlotsPerObs:

    plt.figure(figsize=(30,10))
    plt.subplot(1, 3, 1)
    plt.plot(erprMatrixBaseSubt.T, alpha=0.1)
    plt.xlabel("Time [ms]")
    plt.ylabel("Pupil size [au]")
    plt.title("Per trial")
    plt.plot(np.nanmean(erprMatrixBaseSubt, 0),alpha=1.0, linewidth=2,color='k')


    plt.subplot(1, 3, 2)
    plt.plot(gazeMatrixBaseSubt[:,:,0].T, alpha=0.1)
    plt.xlabel("Time [ms]")
    plt.ylabel("x gaze coordinate [pixels]")
    plt.title("Per trial")
    plt.plot(np.nanmean(gazeMatrixBaseSubt[:,:,0], 0),alpha=1.0, linewidth=2,color='k')
    plt.ylim([np.percentile(gazeMatrixBaseSubt[:,:,0],0.5),np.percentile(gazeMatrixBaseSubt[:,:,0],95.5)])

    plt.subplot(1, 3, 3)
    plt.plot(gazeMatrixBaseSubt[:,:,1].T, alpha=0.1)
    plt.xlabel("Time [ms]")
    plt.ylabel("y gaze coordinate [pixels]")
    plt.title("Per trial")
    plt.plot(np.nanmean(gazeMatrixBaseSubt[:,:,1], 0),alpha=1.0, linewidth=2,color='k')
    plt.ylim([np.percentile(gazeMatrixBaseSubt[:,:,1],0.5),np.percentile(gazeMatrixBaseSubt[:,:,1],99.5)])
    plt.tight_layout()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
