# This file is modified from https://github.com/openbenchmark/BARS/blob/master/ctr_prediction/datasets/Criteo/Criteo_x1/convert_criteo_x1.py
# to crop and convert libsvm data to csv data

import pandas as pd
from pathlib import Path
import gc

headers = ["label", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10",
           "I11", "I12", "I13", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
           "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", 
           "C23", "C24", "C25", "C26"]

data_files = ["train.libsvm", "test.libsvm", "valid.libsvm"]
lengths = [150000, 50000, 30000] 

for f, len in zip(data_files, lengths):
    df = pd.read_csv(f, sep=" ", names=headers, nrows=len)
    for col in headers[1:]:
        if col.startswith("I"):
            df[col] = df[col].apply(lambda x: x.split(':')[-1])
        elif col.startswith("C"):
            df[col] = df[col].apply(lambda x: x.split(':')[0])
    df.to_csv(Path(f).stem + ".csv", index=False)
    del df
    gc.collect()
