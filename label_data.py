import csv
import os
from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta

STATIONS = ["003", "004", "005"]  # station numbers, padded with 3 zeros
DATA_DIR = "DataBinned/HRRR_INL-5minraw_noS35_no-HRRR"
DATA_FILE_NAME = "BinType-avg_BinTS-5_name-Station%s.binned"
OUT_DIR = "DataLabelled"
BIN_DELTA = timedelta(minutes=5)

START_DATE = datetime(2017, 1, 1)
END_DATE = START_DATE + relativedelta(months=2)

# columns in input file
columns = {v: i for i, v in enumerate(['index', 'datetime_bins', 'temp', 'solar', 'speed', 'dir'])}
data = {}  # {station: {date: {k: v}}}

# load data
for station in STATIONS:
    data[station] = {}
    print(os.path.join(DATA_DIR, DATA_FILE_NAME % station))
    with open(os.path.join(DATA_DIR, DATA_FILE_NAME % station), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # skip incomplete rows
            if len(row) != len(columns.keys()) or not row[0].isdigit():
                continue
            date = datetime.strptime(row[columns['datetime_bins']], '%Y-%m-%d %H:%M:%S')
            if date < START_DATE or date > END_DATE:
                continue
            data[station][date] = {v: row[i] for v, i in columns.items()}
            data[station][date]['erroneous'] = 0

# add labels
with open("data_labels.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        station = str(row[2]).zfill(3)
        start = datetime.strptime(row[0].strip(), '%Y-%m-%d %H:%M')
        end = datetime.strptime(row[1].strip(), '%Y-%m-%d %H:%M')
        date = start
        while date <= end:
            if date in data[station]:
                data[station][date]['erroneous'] = 1
            date += BIN_DELTA

# TODO do we want this to all be in one CSV?
# write to file
for station in STATIONS:
    with open(os.path.join(OUT_DIR, '%s.csv' % station), 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['date', 'temp', 'solar', 'speed', 'dir', 'erroneous'])
        for date, values in data[station].items():
            writer.writerow([
                date.isoformat(),
                round(float(values['temp']), 2),
                round(float(values['solar']), 2),
                round(float(values['speed']), 2),
                round(float(values['dir']), 2),
                values['erroneous']
            ])
