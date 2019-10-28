






day = 27
month=2
StartDate = '{}/{}/2017'.format(month,day)
EndDate = '{}/{}/2017'.format(month,day+1)
#EndDate = '2/1/2017'
#StartDate = '3/30/2017'
#EndDate = '4/2/2017'
Station = {}
Station[0] = 7
ReadingType = {}
ReadingType[0] = 'dir'
ReadingType[1] = 'speed'
#ReadingType[2] = 'temp'
#ReadingType[3] = 'solar'
Station = ['Station{:03d}'.format(Station[i]) for i in Station.keys()]
PDR.PlotDataRange(ALLdata,StartDate,EndDate,Station,ReadingType)
    