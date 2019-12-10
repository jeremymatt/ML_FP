

import PlotDataRange as PDR



day = 5
month=2
StartDate = '{}/{}/2017'.format(month,day)
EndDate = '{}/{}/2017'.format(month,day+1)
#EndDate = '2/1/2017'
#EndDate = '3/1/2017'
#StartDate = '3/30/2017'
#EndDate = '4/2/2017'
Station = {}
stations = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,37,38,39,40,42,43,44,45]
stations = [0,1,2,3,4,5,6,7,8]
stations = [0]

for locn in stations:
    Station = {0:locn}
    ReadingType = {}
    ReadingType[0] = 'dir:'
    ReadingType[1] = 'speed:'
    ReadingType[2] = 'temp:'
    ReadingType[3] = 'solar:'
    Station = ['Station{:03d}'.format(Station[i]) for i in Station.keys()]
    PDR.PlotDataRange(ALLdata,StartDate,EndDate,Station,ReadingType,label_prefix = 'label')
    