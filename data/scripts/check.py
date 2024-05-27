import csv
import numpy as np
import os
with open(f"../targets.csv") as f:
    reader = csv.reader(f)
    target_data = [row for row in reader]

print(len(os.listdir("/work/jxliu/e3nn/DOS_net/DosNet_pl/data/dos_raw")))
print(len(os.listdir("/work/jxliu/e3nn/DOS_net/DosNet_pl/data/preprocessed")))
for id in target_data:
    if not os.path.exists(f"/work/jxliu/e3nn/DOS_net/DosNet_pl/data/preprocessed/{id[0]}_feature.csv"):
        print(id[0], " feature")
    if not os.path.exists(f"/work/jxliu/e3nn/DOS_net/DosNet_pl/data/preprocessed/{id[0]}_dosd.csv"):
        print(id[0])
        continue
    with open(f"/work/jxliu/e3nn/DOS_net/DosNet_pl/data/preprocessed/{id[0]}_dosd.csv") as f:
        reader_data = csv.reader(f)
        for row in reader_data:
            for data in row:
                try:
                    if np.isnan(float(data)):
                        print(id[0], "nan")
                        break
                except:
                    print(data)
