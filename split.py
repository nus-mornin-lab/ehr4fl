import pandas as pd
import numpy as np
import os

data_dir = "./data"
out_dir = os.path.join(data_dir, "splitted")

df_all = pd.read_csv(os.path.join(data_dir, "df_eicu.csv"))

splits = df_all.groupby(['hospitalid', 'patientunitstayid']).apply(lambda x: x.assign(split=np.random.choice(['train', 'valid', 'test'], p=[0.8, 0.1, 0.1]))).reset_index(drop=True)
splits['train'] = splits['split'] == 'train'
splits['validation'] = splits['split'] == 'valid'
splits['test'] = splits['split'] == 'test'

train = splits[splits['train']].drop(columns=['train', 'validation', 'test'])
valid = splits[splits['validation']].drop(columns=['train', 'validation', 'test'])
test = splits[splits['test']].drop(columns=['train', 'validation', 'test'])

nrows = {'train': train, 'valid': valid, 'test': test}
nrows = {key: df.groupby('hospitalid').size().reset_index(name='n').sort_values(by='n', ascending=False).head(25) for key, df in nrows.items()}

xs = {key: df.head(25) for key, df in nrows.items()}

intersection = set(xs['train']['hospitalid']) & set(xs['valid']['hospitalid']) & set(xs['test']['hospitalid'])

hospitalids = df_all['hospitalid']

train.to_csv(os.path.join(out_dir, "train_allfeatures_fulldata.csv"), index=False)
valid.to_csv(os.path.join(out_dir, "validation_allfeatures_fulldata.csv"), index=False)
test.to_csv(os.path.join(out_dir, "test_allfeatures_fulldata.csv"), index=False)

for key, df in train.groupby('hospitalid'):
    df.to_csv(os.path.join(out_dir, f"train_allfeatures_hospid_{key}.csv"), index=False)

for key, df in valid.groupby('hospitalid'):
    df.to_csv(os.path.join(out_dir, f"validation_allfeatures_hospid_{key}.csv"), index=False)

for key, df in test.groupby('hospitalid'):
    df.to_csv(os.path.join(out_dir, f"test_allfeatures_hospid_{key}.csv"), index=False)

hospital_ids = [
    73, 122, 188, 199, 208, 243, 248, 252, 264,
    300, 307, 338, 345, 394, 413, 416, 420, 443,
    449, 458
]