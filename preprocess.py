from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from tqdm import tqdm

data_path = Path('../data')
in_path = data_path / 'splitted'
out_simple_path = data_path / 'simple-impute-splitted'


hospitalids = [
  73, 122, 188, 199, 208, 243, 248, 252, 264, 300,
  307, 338, 345, 394, 413, 416, 420, 443, 449, 458
]


first_df = pd.read_csv(in_path/f"train_allfeatures_hospid_{hospitalids[0]}.csv")
first_valid_df = pd.read_csv(in_path/f"validation_allfeatures_hospid_{hospitalids[0]}.csv")


id_columns = ['patientunitstayid']
label_columns = ['death']
other_columns = []

non_feature_columns = id_columns + label_columns + other_columns

binary_features = [
    'is_female',
    'race_black', 'race_hispanic', 'race_asian', 'race_other',
    'electivesurgery'
]

numeric_features = [
    c for c in first_df.columns
    if c not in non_feature_columns + binary_features
]


all_columns = non_feature_columns + numeric_features + binary_features


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])


binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())])


def make_preprocessor(df):
    all_missing = [c for c in df.columns if df[c].isna().all()]

    numeric, binary = tuple(
        list(set(columns).difference(all_missing))
        for columns in (numeric_features, binary_features)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('other', FunctionTransformer(), non_feature_columns),
            ('num', numeric_transformer, numeric),
            ('bin', binary_transformer, binary), *(
                [('na', SimpleImputer(strategy='constant', fill_value=0), all_missing)]
                if all_missing else []
            )])

    columns = non_feature_columns + numeric + binary + all_missing

    return preprocessor, columns


def make_df(arr, columns):
    return pd.DataFrame(
        arr, columns = columns
    ).astype(
        { c: 'int32' for c in non_feature_columns + binary_features }
    )[all_columns]


def process_split(train, valid, test):
    preprocessor, columns = make_preprocessor(train)

    train_arr = preprocessor.fit_transform(train)
    valid_arr = preprocessor.transform(valid)
    test_arr = preprocessor.transform(test)

    return {
        split: make_df(arr, columns)
        for split, arr
        in [('train', train_arr), ('valid', valid_arr), ('test', test_arr)]
    }


def create_federated_split():
    data = (
        (id, {
            'train': pd.read_csv(in_path/f"train_allfeatures_hospid_{id}.csv"),
            'valid': pd.read_csv(in_path/f"validation_allfeatures_hospid_{id}.csv"),
            'test': pd.read_csv(in_path/f"test_allfeatures_hospid_{id}.csv")
        }) for id in hospitalids
    )


    processed_data = [
        (id, process_split(v['train'], v['valid'], v['test']))
        for id, v in data
    ]


    for id, splits in tqdm(processed_data, total=len(hospitalids)):
        splits['train'].to_csv(
            out_simple_path/f"train_allfeatures_hospid_{id}.csv",
            index=False
        )

        splits['valid'].to_csv(
            out_simple_path/f"validation_allfeatures_hospid_{id}.csv",
            index=False
        )
        
        splits['test'].to_csv(
            out_simple_path/f"test_allfeatures_hospid_{id}.csv",
            index=False
        )
    
    return processed_data


def create_centralized_split(inputs):
    splits = {
      s: pd.concat(split[s] for _, split in inputs)
      for s in ('train', 'valid', 'test')
    }

    splits['train'].to_csv(
        out_simple_path/'train_allfeatures_fulldata.csv',
        index=False
    )

    splits['valid'].to_csv(
        out_simple_path/'validation_allfeatures_fulldata.csv',
        index=False
    )
    
    splits['test'].to_csv(
        out_simple_path/'test_allfeatures_fulldata.csv',
        index=False
    )


def main():
    splits = create_federated_split()
    create_centralized_split(splits)


if __name__ == '__main__':
    main()
