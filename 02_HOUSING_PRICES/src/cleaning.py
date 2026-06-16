import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer


COLUMNS_QUAL_NOM = [
    "MSSUBCLASS", "MSZONING", "STREET", "LANDCONTOUR", "UTILITIES",
    "LOTCONFIG", "NEIGHBORHOOD", "CONDITION1", "CONDITION2", "BLDGTYPE",
    "HOUSESTYLE", "ROOFSTYLE", "ROOFMATL", "EXTERIOR1ST", "EXTERIOR2ND",
    "FOUNDATION", "HEATING", "CENTRALAIR", "ELECTRICAL", "GARAGETYPE",
    "GARAGEFINISH", "PAVEDDRIVE", "SALETYPE", "SALECONDITION",
]
    
COLUMNS_QUAL_ORD = [
    "LOTSHAPE", "OVERALLQUAL", "OVERALLCOND", "EXTERQUAL", "EXTERCOND",
    "BSMTQUAL", "BSMTCOND", "BSMTEXPOSURE", "BSMTFINTYPE1", "BSMTFINTYPE2",
    "HEATINGQC", "KITCHENQUAL", "FUNCTIONAL", "GARAGEQUAL", "GARAGECOND",
    "LANDSLOPE",
]

COLUMNS_QUAN_DISC = [
    "YRSOLD", "MOSOLD", "BSMTHALFBATH", "BEDROOMABVGR", "KITCHENABVGR",
    "TOTRMSABVGRD", "GARAGEYRBLT", "HASBASEMENT", "HASGARAGE",
    "HASFIREPLACE", "HASPOOL",
]

COLUMNS_QUAN_CONT = [
    "LOTFRONTAGE", "LOTAREA", "MASVNRAREA", "BSMTFINSF1", "BSMTFINSF2",
    "BSMTUNFSF", "LOWQUALFINSF", "GRLIVAREA", "WOODDECKSF", "ENCLOSEDPORCH",
    "MISCVAL", "TOTALLIVINGSF", "HOUSE_AGE", "REMODEL_AGE", "GARAGE_PROXY",
    "BATHROOM_INDEX", "TOTALPORCHSF",
]

TARGET = "SALEPRICE"

SCALE_COLS = [
    'NEIGHBORHOOD', 'OVERALLQUAL', 'EXTERQUAL', 'BSMTQUAL', 'HEATINGQC', 'GRLIVAREA',
    'KITCHENQUAL', 'TOTRMSABVGRD', 'GARAGEYRBLT', 'GARAGEFINISH', 'ENCLOSEDPORCH',
    'TOTALLIVINGSF', 'HOUSE_AGE', 'REMODEL_AGE', 'GARAGE_PROXY', 'BATHROOM_INDEX', 'TOTALPORCHSF',
]

BINARY_COLS = [
    'HASBASEMENT', 'HASGARAGE', 'HASFIREPLACE', 'FOUNDATION_CBLOCK', 'FOUNDATION_PCONC',
    'EXTERIOR1ST_METALSD', 'EXTERIOR1ST_OTHER', 'EXTERIOR1ST_PLYWOOD', 'EXTERIOR1ST_VINYLSD',
    'EXTERIOR1ST_WD SDNG', 'GARAGETYPE_DETCHD', 'GARAGETYPE_OTHER', 'SALETYPE_CON',
    'SALETYPE_NEW', 'SALETYPE_WD',
]

# Label encodings for nominal columns that require ordinal treatment
_LABEL_ENCODEABLE_NOMINALS = {
    'ROOFSTYLE':     {'GABLE': 1, 'NON-GABLE': 0},
    'NOISE':         {'LOW': 0, 'NORMAL': 1, 'HIGH': 2},
    'MSZONING':      {'RH': 0, 'FV': 1, 'RM': 2, 'RL': 3},
    'SALECONDITION': {'ABNORMAL': 0, 'PARTIAL': 1, 'NORMAL': 2},
    'HOUSESTYLE':    {'1STORY': 0, '1.5STORY': 1, 'SPLIT': 2, '2STORY': 3},
    'GARAGEFINISH':  {'UNF': 0, 'RFN': 1, 'FIN': 2},
}


def normalize_strings(df):
    """Convert object columns to string type and normalize all strings to uppercase."""
    df = df.copy()
    obj_cols = df.select_dtypes(include='object')
    df[obj_cols.columns] = obj_cols.convert_dtypes(convert_string=True)
    df.columns = df.columns.str.upper()
    for col in df.select_dtypes(include='string').columns:
        df[col] = df[col].str.upper()
    return df


def handle_missing_values(df, imputer=None, is_train=True):
    """
    Handle missing values. Always returns (df, imputer, col_info).

    is_train=True : fits KNNImputer, drops high-missing cols and low-missing rows.
    is_train=False: uses fitted imputer, fills low-missing cols with median.
    """
    df = df.copy()
    null_pct = df.isnull().mean() * 100

    high_missing = null_pct[null_pct > 40].index.tolist()
    medium_missing = null_pct[(null_pct <= 40) & (null_pct > 10)].index.tolist()
    low_missing = null_pct[(null_pct > 0) & (null_pct <= 10)].index.tolist()

    col_info = {
        'high_missing': high_missing,
        'medium_missing': medium_missing,
        'low_missing': low_missing,
    }

    df = df.drop(columns=high_missing, errors='ignore')

    if 'GARAGEYRBLT' in df.columns:
        df['GARAGEYRBLT'] = df['GARAGEYRBLT'].fillna(0)

    medium_in_df = [c for c in medium_missing if c in df.columns]
    low_in_df = [c for c in low_missing if c in df.columns]

    if is_train:
        df = df.dropna(subset=low_in_df)
        if medium_in_df:
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            df[medium_in_df] = imputer.fit_transform(df[medium_in_df])
    else:
        for col in low_in_df:
            fill = df[col].mode()[0] if df[col].dtype == 'string' or df[col].dtype == object else df[col].median()
            df[col] = df[col].fillna(fill)
        if medium_in_df and imputer is not None:
            df[medium_in_df] = imputer.transform(df[medium_in_df])

    return df, imputer, col_info


def engineer_features(df):
    """Create engineered features and drop the source columns."""
    df = df.copy()

    df['TOTALLIVINGSF'] = df['TOTALBSMTSF'] + df['1STFLRSF'] + df['2NDFLRSF']
    df['HOUSE_AGE']     = df['YRSOLD'] - df['YEARBUILT']
    df['REMODEL_AGE']   = df['YRSOLD'] - df['YEARREMODADD']
    df['GARAGE_PROXY']  = df['GARAGECARS'] * df['GARAGEAREA']
    df['BATHROOM_INDEX'] = df['FULLBATH'] + 0.5 * df['HALFBATH'] + 0.5 * df['BSMTFULLBATH']
    df['HASBASEMENT']   = (df['TOTALBSMTSF'] > 0).astype(int)
    df['HASGARAGE']     = (df['GARAGEAREA'] > 0).astype(int)
    df['HASFIREPLACE']  = (df['FIREPLACES'] > 0).astype(int)
    df['HASPOOL']       = (df['POOLAREA'] > 0).astype(int)
    df['TOTALPORCHSF']  = df['OPENPORCHSF'] + df['SCREENPORCH'] + df['3SSNPORCH']

    cols_to_drop = [
        'TOTALBSMTSF', '1STFLRSF', '2NDFLRSF',
        'YEARBUILT', 'YEARREMODADD',
        'GARAGECARS', 'GARAGEAREA',
        'FULLBATH', 'HALFBATH', 'BSMTFULLBATH',
        'FIREPLACES', 'POOLAREA',
        'OPENPORCHSF', 'SCREENPORCH', '3SSNPORCH',
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df


def _build_neighborhood_mapping(df):
    """Compute neighborhood → ordinal rank based on mean SalePrice (training only)."""
    nb = (
        df[['NEIGHBORHOOD', TARGET]]
        .groupby('NEIGHBORHOOD')[TARGET]
        .mean()
        .sort_values()
        .reset_index()
    )
    return {row['NEIGHBORHOOD']: i for i, row in nb.iterrows()}


def clean_nominal_qualitative(df, neighborhood_mapping=None, is_train=True):
    """
    Recode and filter nominal qualitative columns.

    is_train=True : computes neighborhood_mapping from SALEPRICE, filters invalid rows.
    is_train=False: uses provided neighborhood_mapping, recodes invalid values instead
                    of dropping rows.

    Returns (df, neighborhood_mapping).
    """
    df = df.copy()

    # Drop outright columns with low predictive value
    cols_drop_outright = [
        'MSSUBCLASS', 'STREET', 'LANDCONTOUR', 'UTILITIES', 'CONDITION2',
        'ROOFMATL', 'EXTERIOR2ND', 'HEATING', 'CENTRALAIR', 'ELECTRICAL',
        'PAVEDDRIVE',
    ]
    df = df.drop(columns=cols_drop_outright, errors='ignore')

    # NEIGHBORHOOD → ordinal by mean SalePrice
    if neighborhood_mapping is None:
        neighborhood_mapping = _build_neighborhood_mapping(df)
    df['NEIGHBORHOOD'] = df['NEIGHBORHOOD'].map(neighborhood_mapping)

    # MSZONING
    if 'MSZONING' in df.columns:
        df.loc[df['MSZONING'] == 'C (ALL)', 'MSZONING'] = np.nan
        if is_train:
            df = df.dropna(subset=['MSZONING'])
        else:
            df['MSZONING'] = df['MSZONING'].fillna(df['MSZONING'].mode()[0])

    # LOTCONFIG
    if 'LOTCONFIG' in df.columns:
        df['LOTCONFIG'] = df['LOTCONFIG'].replace({
            'INSIDE': 'SINGLE_FRONTAGE', 'CULDSAC': 'SINGLE_FRONTAGE',
            'CORNER': 'MULTIPLE_FRONTAGE', 'FR2': 'MULTIPLE_FRONTAGE', 'FR3': 'MULTIPLE_FRONTAGE',
        })

    # CONDITION1 → NOISE
    if 'CONDITION1' in df.columns:
        df = df.rename(columns={'CONDITION1': 'NOISE'})
        df['NOISE'] = df['NOISE'].replace({
            'ARTERY': 'HIGH', 'FEEDR': 'HIGH', 'RRNN': 'HIGH',
            'RRAN': 'HIGH', 'RRNE': 'HIGH', 'RRAE': 'HIGH',
            'POSN': 'LOW', 'POSA': 'LOW', 'NORM': 'NORMAL',
        })

    # BLDGTYPE
    if 'BLDGTYPE' in df.columns:
        df['BLDGTYPE'] = df['BLDGTYPE'].replace({
            'TWNHSE': 'TOWNHOUSE', 'TWNHSI': 'TOWNHOUSE', 'TWNHS': 'TOWNHOUSE',
            'DUPLEX': 'MULTI-FAMILY', '2FMCON': 'MULTI-FAMILY',
        })

    # HOUSESTYLE
    if 'HOUSESTYLE' in df.columns:
        df['HOUSESTYLE'] = df['HOUSESTYLE'].replace({
            '1STORY': '1STORY', '2STORY': '2STORY',
            '1.5FIN': '1.5STORY', '1.5UNF': '1.5STORY',
            '2.5FIN': '2STORY', '2.5UNF': '2STORY',
            'SLVL': 'SPLIT', 'SFOYER': 'SPLIT',
        })

    # ROOFSTYLE
    if 'ROOFSTYLE' in df.columns:
        df['ROOFSTYLE'] = df['ROOFSTYLE'].replace({
            'GABLE': 'GABLE',
            'HIP': 'NON-GABLE', 'FLAT': 'NON-GABLE', 'GAMBREL': 'NON-GABLE',
            'MANSARD': 'NON-GABLE', 'SHED': 'NON-GABLE',
        })

    # EXTERIOR1ST
    if 'EXTERIOR1ST' in df.columns:
        df['EXTERIOR1ST'] = df['EXTERIOR1ST'].replace({
            'VINYLSD': 'VINYLSD', 'HDBOARD': 'HDBOARD', 'METALSD': 'METALSD',
            'WD SDNG': 'WD SDNG', 'PLYWOOD': 'PLYWOOD',
            'CEMNTBD': 'OTHER', 'BRKFACE': 'OTHER', 'STUCCO': 'OTHER',
            'WDSHING': 'OTHER', 'ASBSHNG': 'OTHER', 'STONE': 'OTHER',
            'BRKCOMM': 'OTHER', 'IMSTUCC': 'OTHER', 'CBLOCK': 'OTHER',
        })

    # FOUNDATION: only common types
    valid_found = ['PCONC', 'CBLOCK', 'BRKTIL']
    if 'FOUNDATION' in df.columns:
        if is_train:
            df = df[df['FOUNDATION'].isin(valid_found)]
        else:
            df.loc[~df['FOUNDATION'].isin(valid_found), 'FOUNDATION'] = 'CBLOCK'

    # GARAGETYPE
    if 'GARAGETYPE' in df.columns:
        df['GARAGETYPE'] = df['GARAGETYPE'].replace({
            'ATTCHD': 'ATTCHD', 'DETCHD': 'DETCHD',
            'BUILTIN': 'OTHER', 'BASMENT': 'OTHER', 'CARPORT': 'OTHER', '2TYPES': 'OTHER',
        })

    # SALETYPE
    if 'SALETYPE' in df.columns:
        df['SALETYPE'] = df['SALETYPE'].replace({
            'WD': 'WD', 'NEW': 'NEW', 'COD': 'COD',
            'CONLD': 'CON', 'CONLI': 'CON', 'CWD': 'WD', 'CONLW': 'CON',
            'OTH': 'OTH',
        })
        if is_train:
            df = df[df['SALETYPE'] != 'OTH']
        else:
            df.loc[df['SALETYPE'] == 'OTH', 'SALETYPE'] = 'WD'

    # SALECONDITION
    if 'SALECONDITION' in df.columns:
        valid_conds = ['NORMAL', 'PARTIAL', 'ABNORML']
        if is_train:
            df = df[df['SALECONDITION'].isin(valid_conds)]
        else:
            df.loc[~df['SALECONDITION'].isin(valid_conds), 'SALECONDITION'] = 'NORMAL'
        df['SALECONDITION'] = df['SALECONDITION'].replace('ABNORML', 'ABNORMAL')

    return df, neighborhood_mapping


def drop_low_corr_nominal_dummy(df, threshold=0.4, columns_to_drop=None):
    """
    Drop nominal columns with low grouped correlation to SALEPRICE via dummy analysis.

    If columns_to_drop is None, compute from training data. Else apply provided list.
    Returns (df, columns_to_drop).
    """
    df = df.copy()
    cols_for_analysis = [
        'FOUNDATION', 'EXTERIOR1ST', 'GARAGETYPE', 'BLDGTYPE',
        'LOTCONFIG', 'ROOFSTYLE', 'SALETYPE',
    ]

    if columns_to_drop is None:
        present = [c for c in cols_for_analysis if c in df.columns]
        dummy_df = pd.get_dummies(df[present], drop_first=False, dtype=int)
        dummy_df[TARGET] = df[TARGET]

        corr = dummy_df.corr()[TARGET].abs()
        corr.index = corr.index.str.split('_', n=0).str[0]
        corr_grouped = corr.groupby(corr.index).sum()

        mask = corr_grouped < threshold
        columns_to_drop = mask[mask].index.tolist()

    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df, columns_to_drop


def encode_dummies_nominal(df, columns_to_dummy=None):
    """Apply one-hot encoding to specified nominal columns."""
    df = df.copy()
    if columns_to_dummy is None:
        columns_to_dummy = ['FOUNDATION', 'EXTERIOR1ST', 'GARAGETYPE', 'SALETYPE']
    for col in [c for c in columns_to_dummy if c in df.columns]:
        df = pd.get_dummies(df, columns=[col], drop_first=True, dtype='uint8')
    return df


def drop_low_corr_nominal_label(df, threshold=0.4, columns_to_drop=None):
    """
    Drop label-encodeable nominal columns with low correlation to SALEPRICE.

    Temporarily applies label encodings for the correlation analysis only.
    Returns (df, columns_to_drop).
    """
    df = df.copy()
    candidates = [c for c in _LABEL_ENCODEABLE_NOMINALS if c in df.columns]

    if columns_to_drop is None:
        analysis = df[candidates].copy()
        for col, mapping in _LABEL_ENCODEABLE_NOMINALS.items():
            if col in analysis.columns:
                analysis[col] = analysis[col].map(mapping)
        analysis[TARGET] = df[TARGET]

        corr = analysis.corr()[TARGET].abs()
        mask = corr < threshold
        columns_to_drop = mask[mask].index.drop(TARGET, errors='ignore').tolist()

    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df, columns_to_drop


def encode_label_nominal(df):
    """Apply label encoding to surviving nominal columns."""
    df = df.copy()
    for col, mapping in _LABEL_ENCODEABLE_NOMINALS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


def encode_ordinal_qualitative(df):
    """Apply ordinal encodings to all ordinal qualitative columns present in df."""
    df = df.copy()

    col_mappings = {
        'LOTSHAPE':    {'IR3': 0, 'IR2': 1, 'IR1': 2, 'REG': 3},
        'EXTERQUAL':   {'PO': 0, 'FA': 1, 'TA': 2, 'GD': 3, 'EX': 4},
        'EXTERCOND':   {'PO': 0, 'FA': 1, 'TA': 2, 'GD': 3, 'EX': 4},
        'HEATINGQC':   {'PO': 0, 'FA': 1, 'TA': 2, 'GD': 3, 'EX': 4},
        'KITCHENQUAL': {'PO': 0, 'FA': 1, 'TA': 2, 'GD': 3, 'EX': 4},
        'GARAGEQUAL':  {'PO': 0, 'FA': 1, 'TA': 2, 'GD': 3, 'EX': 4},
        'GARAGECOND':  {'PO': 0, 'FA': 1, 'TA': 2, 'GD': 3, 'EX': 4},
        'BSMTQUAL':    {'NA': 0, 'PO': 1, 'FA': 2, 'TA': 3, 'GD': 4, 'EX': 5},
        'BSMTCOND':    {'NA': 0, 'PO': 1, 'FA': 2, 'TA': 3, 'GD': 4, 'EX': 5},
        'BSMTEXPOSURE':{'NA': 0, 'NO': 1, 'MN': 2, 'AV': 3, 'GD': 4},
        'BSMTFINTYPE1':{'NA': 0, 'UNF': 1, 'LWQ': 2, 'REC': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'BSMTFINTYPE2':{'NA': 0, 'UNF': 1, 'LWQ': 2, 'REC': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'FUNCTIONAL':  {'SAL': 0, 'SEV': 1, 'MAJ2': 2, 'MAJ1': 3, 'MOD': 4, 'MIN2': 5, 'MIN1': 6, 'TYP': 7},
        'LANDSLOPE':   {'SEV': 0, 'MOD': 1, 'GTL': 2},
    }

    for col, mapping in col_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df


def drop_low_corr_ordinal(df, threshold=0.4, columns_to_drop=None):
    """Drop ordinal columns with abs correlation to SALEPRICE below threshold."""
    df = df.copy()
    ord_cols = [c for c in COLUMNS_QUAL_ORD if c in df.columns]

    if columns_to_drop is None:
        corr = df[ord_cols + [TARGET]].corr()[TARGET].abs()
        mask = corr < threshold
        columns_to_drop = mask[mask].index.drop(TARGET, errors='ignore').tolist()

    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df, columns_to_drop


def drop_low_corr_discrete(df, threshold=0.4, columns_to_drop=None):
    """Drop discrete quantitative columns with abs correlation to SALEPRICE below threshold."""
    df = df.copy()
    disc_cols = [c for c in COLUMNS_QUAN_DISC if c in df.columns]

    if columns_to_drop is None:
        corr = df[disc_cols + [TARGET]].corr()[TARGET].abs()
        mask = corr < threshold
        columns_to_drop = mask[mask].index.drop(TARGET, errors='ignore').tolist()

    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df, columns_to_drop


def apply_log_transform(df, cols_to_log=None):
    """
    Apply log1p to skewed continuous columns.
    If cols_to_log is None, detect skewed columns from data (training mode).
    Returns (df, cols_to_log).
    """
    df = df.copy()
    cont_cols = [c for c in COLUMNS_QUAN_CONT + [TARGET] if c in df.columns]

    if cols_to_log is None:
        skewness = df[cont_cols].skew()
        cols_to_log = skewness[(skewness > 1) | (skewness < -1)].index.tolist()

    for col in cols_to_log:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    return df, cols_to_log


def remove_outliers(df, max_total_loss=0.15):
    """
    Remove outlier rows from continuous columns using a budgeted IQR strategy.
    Most aggressive columns are processed first; stops when the row-loss budget is used.
    Only for training data — do not apply to test.
    """
    df = df.copy()
    cont_cols = [c for c in COLUMNS_QUAN_CONT + [TARGET] if c in df.columns]
    n_original = len(df)

    candidates = [
        (0.25, 0.75), (0.20, 0.80), (0.15, 0.85),
        (0.10, 0.90), (0.05, 0.95), (0.02, 0.98), (0.01, 0.99),
    ]

    # Rank columns by outlier severity using the tightest IQR range
    severity = {}
    for col in cont_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)
        severity[col] = 1 - mask.sum() / len(df)

    for col in sorted(severity, key=severity.get, reverse=True):
        if 1 - len(df) / n_original >= max_total_loss:
            break

        for q1_pct, q3_pct in candidates:
            q1 = df[col].quantile(q1_pct)
            q3 = df[col].quantile(q3_pct)
            iqr = q3 - q1
            mask = (df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)

            if (1 - mask.sum() / n_original) <= max_total_loss:
                df = df[mask]
                break

    return df


def drop_low_corr_continuous(df, threshold=0.4, columns_to_drop=None):
    """Drop continuous quantitative columns with abs correlation to SALEPRICE below threshold."""
    df = df.copy()
    cont_cols = [c for c in COLUMNS_QUAN_CONT + [TARGET] if c in df.columns]

    if columns_to_drop is None:
        corr = df[cont_cols].corr()[TARGET].abs()
        mask = corr < threshold
        columns_to_drop = mask[mask].index.drop(TARGET, errors='ignore').tolist()

    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df, columns_to_drop


def build_preprocessor(df, preprocessor=None):
    """
    Build or apply a ColumnTransformer:
    - StandardScaler on SCALE_COLS (continuous/ordinal)
    - passthrough on BINARY_COLS (binary/dummy)
    - remainder='drop' removes TARGET and ID automatically

    Training (preprocessor=None): fit_transform. Test: transform only.
    Returns (df_transformed, preprocessor).
    """
    df = df.copy()

    y = df[TARGET].copy() if TARGET in df.columns else None
    feature_df = df.drop(columns=[TARGET], errors='ignore')

    scale_present = [c for c in SCALE_COLS if c in feature_df.columns]
    binary_present = [c for c in BINARY_COLS if c in feature_df.columns]

    if preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('scale', preprocessing.StandardScaler(), scale_present),
                ('binary', 'passthrough', binary_present),
            ],
            remainder='drop',
        )
        arr = preprocessor.fit_transform(feature_df)
    else:
        arr = preprocessor.transform(feature_df)

    transformed = pd.DataFrame(arr, columns=scale_present + binary_present, index=feature_df.index)

    if y is not None:
        transformed[TARGET] = y.values

    return transformed, preprocessor


def clean_train(df):
    """Full cleaning pipeline for training data. Returns (df_clean, state)."""
    state = {}

    df = normalize_strings(df)

    df, state['imputer'], _ = handle_missing_values(df, is_train=True)

    df = engineer_features(df)

    df, state['neighborhood_mapping'] = clean_nominal_qualitative(df, is_train=True)

    df, state['dropped_nom_dummy'] = drop_low_corr_nominal_dummy(df)
    df = encode_dummies_nominal(df)

    df, state['dropped_nom_label'] = drop_low_corr_nominal_label(df)
    df = encode_label_nominal(df)

    df = encode_ordinal_qualitative(df)
    df, state['dropped_ord'] = drop_low_corr_ordinal(df)

    df, state['dropped_disc'] = drop_low_corr_discrete(df)

    df, state['cols_to_log'] = apply_log_transform(df)
    df = remove_outliers(df)
    df, state['dropped_cont'] = drop_low_corr_continuous(df)

    df, state['preprocessor'] = build_preprocessor(df)
    df = df.reset_index(drop=True)

    return df, state


def clean_test(df, state):
    """Apply the cleaning pipeline to test data using the state derived from training."""
    df = normalize_strings(df)
    test_ids = df['ID'].copy() if 'ID' in df.columns else None
    df, _, _ = handle_missing_values(df, imputer=state['imputer'], is_train=False)
    df = engineer_features(df)

    df, _ = clean_nominal_qualitative(df, neighborhood_mapping=state['neighborhood_mapping'], is_train=False)

    df = df.drop(columns=state['dropped_nom_dummy'], errors='ignore')
    df = encode_dummies_nominal(df)

    df = df.drop(columns=state['dropped_nom_label'], errors='ignore')
    df = encode_label_nominal(df)

    df = encode_ordinal_qualitative(df)
    df = df.drop(columns=state['dropped_ord'] + state['dropped_disc'], errors='ignore')

    df, _ = apply_log_transform(df, cols_to_log=state['cols_to_log'])
    df = df.drop(columns=state['dropped_cont'], errors='ignore')

    df, _ = build_preprocessor(df, preprocessor=state['preprocessor'])
    df = df.reset_index(drop=True)

    if test_ids is not None:
        df.insert(0, 'ID', test_ids.values)

    return df
