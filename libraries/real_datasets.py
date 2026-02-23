import numpy as np
import pandas as pd

def _train_test_split_idx(n, test_frac=0.25, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = int(test_frac * n)
    return idx[n_test:], idx[:n_test]

def load_titanic_raw():
    # Try seaborn first (common in Colab)
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        return df
    except Exception:
        pass

    # Fallback: OpenML
    try:
        from sklearn.datasets import fetch_openml
        Xy = fetch_openml("titanic", version=1, as_frame=True)
        df = Xy.frame
        return df
    except Exception as e:
        raise RuntimeError(
            "Could not load Titanic. Install seaborn or sklearn, or provide a local CSV.\n"
            "Try: pip install seaborn scikit-learn\n"
            f"Original error: {e}"
        )

def titanic_source_target_split(df: pd.DataFrame, source_embarked="S"):
    # Keep columns that exist in seaborn titanic
    cols = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "alone"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()

    # Drop missing target
    df = df.dropna(subset=["survived"])

    # Define domains by embarked (S = source, others = target)
    if "embarked" not in df.columns:
        raise RuntimeError("Titanic dataset missing 'embarked' column; cannot domain-split.")

    src_df = df[df["embarked"] == source_embarked].copy()
    tgt_df = df[df["embarked"] != source_embarked].copy()

    # If split is too small, fallback to pclass domain split
    if len(src_df) < 200 or len(tgt_df) < 200:
        if "pclass" in df.columns:
            src_df = df[df["pclass"] == 3].copy()
            tgt_df = df[df["pclass"] != 3].copy()

    return src_df, tgt_df

def preprocess_tabular(train_df, test_df, target_col):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(np.float32).to_numpy()

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(np.float32).to_numpy()

    num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )

    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)

    # Convert sparse -> dense (safe for small datasets)
    try:
        Xtr = Xtr.toarray()
        Xte = Xte.toarray()
    except Exception:
        pass

    return Xtr.astype(np.float32), y_train, Xte.astype(np.float32), y_test, pre

def load_titanic_logistic(seed=0, test_frac=0.25, source_embarked="S"):
    df = load_titanic_raw()
    src_df, tgt_df = titanic_source_target_split(df, source_embarked=source_embarked)

    # split each domain into train/test
    src_tr_idx, src_te_idx = _train_test_split_idx(len(src_df), test_frac=test_frac, seed=seed)
    tgt_tr_idx, tgt_te_idx = _train_test_split_idx(len(tgt_df), test_frac=test_frac, seed=seed + 1)

    src_train, src_test = src_df.iloc[src_tr_idx], src_df.iloc[src_te_idx]
    tgt_train, tgt_test = tgt_df.iloc[tgt_tr_idx], tgt_df.iloc[tgt_te_idx]

    # Fit ONE shared preprocessor on union(train) to keep feature spaces consistent
    union_train = pd.concat([src_train, tgt_train], axis=0)
    X_union_tr, y_union_tr, _, _, pre = preprocess_tabular(union_train, union_train, target_col="survived")
    # Re-transform each split using that preprocessor
    def transform(df_):
        X = df_.drop(columns=["survived"])
        y = df_["survived"].astype(np.float32).to_numpy()
        Xt = pre.transform(X)
        try:
            Xt = Xt.toarray()
        except Exception:
            pass
        return Xt.astype(np.float32), y

    Xs_tr, ys_tr = transform(src_train)
    Xs_te, ys_te = transform(src_test)
    Xt_tr, yt_tr = transform(tgt_train)
    Xt_te, yt_te = transform(tgt_test)

    return (Xs_tr, ys_tr, Xs_te, ys_te), (Xt_tr, yt_tr, Xt_te, yt_te)


def load_breast_cancer_logistic(seed=0, test_frac=0.25):
    """
    Breast Cancer Wisconsin — logistic regression (predict malignant/benign).

    Domain split by tumor size: source = smaller tumors (mean radius < median),
    target = larger tumors.  This creates a natural covariate shift.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()

    # Domain split by mean radius (feature 0)
    median_radius = df["mean radius"].median()
    src_df = df[df["mean radius"] <= median_radius].copy()
    tgt_df = df[df["mean radius"] > median_radius].copy()

    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]

    src_tr_idx, src_te_idx = _train_test_split_idx(len(src_df), test_frac=test_frac, seed=seed)
    tgt_tr_idx, tgt_te_idx = _train_test_split_idx(len(tgt_df), test_frac=test_frac, seed=seed + 1)

    src_train, src_test = src_df.iloc[src_tr_idx], src_df.iloc[src_te_idx]
    tgt_train, tgt_test = tgt_df.iloc[tgt_tr_idx], tgt_df.iloc[tgt_te_idx]

    union_train = pd.concat([src_train, tgt_train], axis=0)
    sc = StandardScaler().fit(union_train[feature_cols].to_numpy())

    def xy(split):
        X = sc.transform(split[feature_cols].to_numpy()).astype(np.float32)
        y = split[target_col].to_numpy().astype(np.float32)
        return X, y

    Xs_tr, ys_tr = xy(src_train)
    Xs_te, ys_te = xy(src_test)
    Xt_tr, yt_tr = xy(tgt_train)
    Xt_te, yt_te = xy(tgt_test)

    return (Xs_tr, ys_tr, Xs_te, ys_te), (Xt_tr, yt_tr, Xt_te, yt_te)


def load_wine_linear(seed=0, test_frac=0.25):
    """
    Wine Quality — linear regression (predict quality score).

    Domain split by color: source = red wine, target = white wine.
    Uses the UCI Wine Quality dataset via sklearn.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler

    # Load red wine
    red = fetch_openml("wine-quality-red", version=1, as_frame=True, parser="auto")
    red_df = red.frame.copy()

    # Load white wine
    white = fetch_openml("wine-quality-white", version=1, as_frame=True, parser="auto")
    white_df = white.frame.copy()

    # OpenML uses varying column names across versions/platforms.
    # Both datasets have 11 features + 1 target.  Standardize to common names.
    canonical_features = [
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "pH", "sulphates", "alcohol",
    ]
    target_col = "quality"

    def _normalize_wine_cols(df):
        """Rename columns to canonical names regardless of OpenML format."""
        cols = list(df.columns)
        # Find the target column (last column, often 'class', 'Class', or 'quality')
        tgt_candidates = [c for c in cols if c.lower() in ("class", "quality")]
        if tgt_candidates:
            tgt_name = tgt_candidates[0]
        else:
            tgt_name = cols[-1]  # fallback: last column is target
        feat_cols = [c for c in cols if c != tgt_name]

        rename_map = {tgt_name: target_col}
        for i, fc in enumerate(feat_cols):
            if i < len(canonical_features):
                rename_map[fc] = canonical_features[i]
        df = df.rename(columns=rename_map)
        df[target_col] = df[target_col].astype(float)
        return df

    red_df = _normalize_wine_cols(red_df)
    white_df = _normalize_wine_cols(white_df)

    feature_cols = canonical_features

    src_df = red_df
    tgt_df = white_df

    src_tr_idx, src_te_idx = _train_test_split_idx(len(src_df), test_frac=test_frac, seed=seed)
    tgt_tr_idx, tgt_te_idx = _train_test_split_idx(len(tgt_df), test_frac=test_frac, seed=seed + 1)

    src_train, src_test = src_df.iloc[src_tr_idx], src_df.iloc[src_te_idx]
    tgt_train, tgt_test = tgt_df.iloc[tgt_tr_idx], tgt_df.iloc[tgt_te_idx]

    union_train = pd.concat([src_train, tgt_train], axis=0)
    sc = StandardScaler().fit(union_train[feature_cols].to_numpy())

    def xy(split):
        X = sc.transform(split[feature_cols].to_numpy()).astype(np.float32)
        y = split[target_col].to_numpy().astype(np.float32)
        return X, y

    Xs_tr, ys_tr = xy(src_train)
    Xs_te, ys_te = xy(src_test)
    Xt_tr, yt_tr = xy(tgt_train)
    Xt_te, yt_te = xy(tgt_test)

    return (Xs_tr, ys_tr, Xs_te, ys_te), (Xt_tr, yt_tr, Xt_te, yt_te)


def load_california_housing_linear(seed=0, test_frac=0.25):
    """
    California Housing — linear regression (predict median house value).

    Domain split by latitude: source = Northern CA (Bay Area, Sacramento),
    target = Southern CA (LA, San Diego).  This creates a natural covariate
    shift — housing patterns differ but the feature-price relationship
    has strong overlap, making it ideal for transfer learning.

    20,640 samples, 8 features.
    """
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler

    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    # data.frame already has MedHouseVal; rename it to "target"
    df = df.rename(columns={"MedHouseVal": "target"})

    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]

    # Domain split: North CA (latitude > median) vs South CA
    median_lat = df["Latitude"].median()
    src_df = df[df["Latitude"] >= median_lat].copy()
    tgt_df = df[df["Latitude"] < median_lat].copy()

    src_tr_idx, src_te_idx = _train_test_split_idx(len(src_df), test_frac=test_frac, seed=seed)
    tgt_tr_idx, tgt_te_idx = _train_test_split_idx(len(tgt_df), test_frac=test_frac, seed=seed + 1)

    src_train, src_test = src_df.iloc[src_tr_idx], src_df.iloc[src_te_idx]
    tgt_train, tgt_test = tgt_df.iloc[tgt_tr_idx], tgt_df.iloc[tgt_te_idx]

    # Shared scaler fit on union of training data
    union_train = pd.concat([src_train, tgt_train], axis=0)
    sc = StandardScaler().fit(union_train[feature_cols].to_numpy())

    def xy(split):
        X = sc.transform(split[feature_cols].to_numpy()).astype(np.float32)
        y = split[target_col].to_numpy().astype(np.float32)
        return X, y

    Xs_tr, ys_tr = xy(src_train)
    Xs_te, ys_te = xy(src_test)
    Xt_tr, yt_tr = xy(tgt_train)
    Xt_te, yt_te = xy(tgt_test)

    return (Xs_tr, ys_tr, Xs_te, ys_te), (Xt_tr, yt_tr, Xt_te, yt_te)