"""Curated suite of small, non-linear OpenML datasets for evaluating TabPFN and its distillations.

The suite contains 61 datasets (51 classification + 10 regression), almost all with <= 1000 rows
(the canonical Titanic set is the sole exception at 1309) and known non-linear structure (XOR / multiplicative interactions, multi-class geometry, signal / physics
data). This is exactly TabPFN's regime, and it is where a strong teacher / student pulls clearly ahead
of a linear baseline -- so distillation gains are easy to see.

Everything is fetched through ``sklearn.datasets.fetch_openml(data_id=...)`` (no extra dependency, with
built-in on-disk caching). ``OpenMLBenchmark`` loads any dataset into model-ready ``np.float32`` arrays
and runs a repeated train/test-split evaluation, reusing the project's conventions
(``StratifiedShuffleSplit`` / ``ShuffleSplit`` with ``random_state=0``, ``StandardScaler`` for students,
accuracy and ROC AUC for classification and ``r2_score`` / RMSE for regression). ROC AUC is the metric
where TabPFN's edge over tree baselines is clearest; plain accuracy thresholds away that advantage.

Example
-------
>>> from benchmark_datasets import OpenMLBenchmark
>>> from tabpfn import TabPFNClassifier, TabPFNRegressor
>>> bench = OpenMLBenchmark()
>>> bench.list()                                          # registry as a DataFrame
>>> ds = bench.load("sonar")                              # -> LoadedDataset(X, y, ...)
>>> bench.evaluate(                                       # teacher across all classification sets
...     lambda task: TabPFNClassifier(), task="classification")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

Task = Literal["classification", "regression"]

RANDOM_STATE = 0


def _to_class_labels(pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Snap predictions to valid class labels.

    True classifiers already return labels in ``classes`` and pass through unchanged. Distillation
    students that regress onto soft targets (e.g. an ``MLPRegressor`` trained on ``predict_proba``)
    return continuous values, which are mapped to the nearest class label -- the generalisation of the
    notebook's ``predict(...) > 0.5``.
    """
    pred = np.asarray(pred)
    classes = np.asarray(classes)
    if np.issubdtype(pred.dtype, np.floating) and not np.all(np.isin(pred, classes)):
        nearest = np.abs(pred.reshape(-1, 1) - classes.reshape(1, -1)).argmin(axis=1)
        return classes[nearest]
    return pred


def _class_scores(model: object, X: np.ndarray) -> np.ndarray:
    """Return soft scores for ROC AUC: ``predict_proba`` > ``decision_function`` > raw ``predict``.

    The raw-``predict`` fallback is exactly what a distillation regressor outputs (its predicted
    probability), so AUC is well defined for those students too.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    return np.asarray(model.predict(X))


def _roc_auc(y_true: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> float:
    """ROC AUC for binary (positive-class score) or multiclass (one-vs-rest macro).

    Returns ``nan`` when AUC is undefined for this split (e.g. a class missing from the test fold, or
    only 1-D scores available for a multiclass problem).
    """
    scores = np.asarray(scores)
    try:
        if len(classes) == 2:
            s = scores[:, 1] if scores.ndim == 2 else scores
            return float(roc_auc_score(y_true, s))
        if scores.ndim == 2 and scores.shape[1] == len(classes):
            return float(
                roc_auc_score(y_true, scores, multi_class="ovr", average="macro", labels=classes)
            )
        return float("nan")
    except ValueError:
        return float("nan")


@dataclass(frozen=True)
class DatasetSpec:
    """A single OpenML dataset in the benchmark registry."""

    name: str
    data_id: int
    task: Task
    n_rows: int
    note: str = ""
    drop_cols: tuple[str, ...] = ()  # columns to drop before encoding (leaks / high-cardinality ids)


# 51 small, non-linear classification datasets (all <= 1000 rows except titanic at 1309). data_id /
# row counts verified
# against OpenML (data/qualities + data/{id} metadata endpoints).
CLASSIFICATION_DATASETS: list[DatasetSpec] = [
    DatasetSpec("tic-tac-toe", 50, "classification", 958, "XOR-like win patterns, pure interaction"),
    DatasetSpec("monks-problems-2", 334, "classification", 601, "synthetic XOR / parity target"),
    DatasetSpec("sonar", 40, "classification", 208, "60 correlated sonar bands, non-linear boundary"),
    DatasetSpec("ionosphere", 59, "classification", 351, "radar signal, non-linear"),
    DatasetSpec("vehicle", 54, "classification", 846, "4-class silhouette geometry"),
    DatasetSpec("wdbc", 1510, "classification", 569, "breast cancer, non-linear feature interactions"),
    DatasetSpec("diabetes", 37, "classification", 768, "Pima diabetes, classic non-linear medical"),
    DatasetSpec("ilpd", 1480, "classification", 583, "Indian liver patient, non-linear medical"),
    DatasetSpec("balance-scale", 11, "classification", 625, "target is a product (distance x weight)"),
    DatasetSpec("blood-transfusion", 1464, "classification", 748, "non-linear recency / frequency"),
    DatasetSpec(
        "titanic", 40945, "classification", 1309,
        "survival from passenger features, non-linear interactions (sex x class x age)",
        drop_cols=("name", "ticket", "cabin", "home.dest", "boat", "body"),
    ),
    # --- 40 additional small classification datasets ---
    DatasetSpec("heart-statlog", 53, "classification", 270, "Statlog heart disease, non-linear medical"),
    DatasetSpec("glass", 41, "classification", 214, "6-class glass type from oxide composition"),
    DatasetSpec("wine", 187, "classification", 178, "3-class cultivars, multiplicative chem interactions"),
    DatasetSpec("zoo", 62, "classification", 101, "7-class animal taxonomy from binary traits"),
    DatasetSpec("hepatitis", 55, "classification", 155, "survival prediction, non-linear medical"),
    DatasetSpec("lymph", 10, "classification", 148, "4-class lymphography, non-linear"),
    DatasetSpec("tae", 48, "classification", 151, "3-class teaching-assistant evaluation"),
    DatasetSpec("haberman", 43, "classification", 306, "breast-cancer survival, non-linear"),
    DatasetSpec("vote", 56, "classification", 435, "congressional votes, interaction-heavy"),
    DatasetSpec("monks-problems-1", 333, "classification", 556, "synthetic logical-rule target"),
    DatasetSpec("monks-problems-3", 335, "classification", 554, "synthetic logical rule with noise"),
    DatasetSpec("planning-relax", 1490, "classification", 182, "EEG planning vs relax, non-linear signal"),
    DatasetSpec("credit-approval", 29, "classification", 690, "credit approval, mixed-type non-linear"),
    DatasetSpec("breast-w", 15, "classification", 699, "Wisconsin breast cancer, non-linear medical"),
    DatasetSpec("breast-cancer", 13, "classification", 286, "recurrence, categorical interactions"),
    DatasetSpec("credit-g", 31, "classification", 1000, "German credit risk, non-linear"),
    DatasetSpec("ecoli", 39, "classification", 336, "8-class protein localisation, non-linear"),
    DatasetSpec("flags", 285, "classification", 194, "8-class religion from flag features"),
    DatasetSpec("cleveland", 786, "classification", 303, "heart disease, non-linear medical"),
    DatasetSpec("colic", 27, "classification", 368, "horse colic surgery, non-linear w/ missing data"),
    DatasetSpec("biomed", 481, "classification", 209, "biomedical screening, non-linear"),
    DatasetSpec("analcatdata_authorship", 458, "classification", 841, "4-class authorship from word freqs"),
    DatasetSpec("analcatdata_lawsuit", 450, "classification", 264, "imbalanced layoff classification"),
    DatasetSpec("climate-crashes", 1467, "classification", 540, "rare simulation crashes, non-linear"),
    DatasetSpec("acute-inflammations", 1455, "classification", 120, "rule-based diagnosis, logical interactions"),
    DatasetSpec("breast-tissue", 1465, "classification", 106, "6-class impedance tissue geometry"),
    DatasetSpec("fertility", 1473, "classification", 100, "fertility diagnosis, non-linear medical"),
    DatasetSpec("corral", 40669, "classification", 160, "synthetic correlated/irrelevant feats (XOR core)"),
    DatasetSpec("australian", 40981, "classification", 690, "Australian credit, mixed-type non-linear"),
    DatasetSpec("conference_attendance", 41538, "classification", 246, "attendance prediction, interactions"),
    DatasetSpec("autoUniv-au7-700", 1553, "classification", 700, "synthetic 3-class, non-linear"),
    DatasetSpec("autoUniv-au6-400", 1551, "classification", 400, "synthetic 8-class, non-linear geometry"),
    DatasetSpec("diabetes-risk", 46733, "classification", 520, "early diabetes symptoms, non-linear"),
    DatasetSpec("ar1", 1059, "classification", 121, "software defect prediction, non-linear"),
    DatasetSpec("ar4", 1061, "classification", 107, "software defect prediction, non-linear"),
    DatasetSpec("backache", 463, "classification", 180, "backache risk factors, non-linear medical"),
    DatasetSpec("datatrieve", 1075, "classification", 130, "software fault prediction, non-linear"),
    DatasetSpec("profb", 470, "classification", 672, "pro-football outcomes, non-linear"),
    DatasetSpec("kc2", 1063, "classification", 522, "NASA software defects, non-linear"),
    DatasetSpec("megawatt1", 1442, "classification", 253, "software defect prediction, non-linear"),
]

# 10 small, non-linear regression datasets (all <= 1000 rows).
REGRESSION_DATASETS: list[DatasetSpec] = [
    DatasetSpec("autoMpg", 196, "regression", 398, "non-linear mpg vs weight / horsepower"),
    DatasetSpec("machine_cpu", 230, "regression", 209, "non-linear CPU performance"),
    DatasetSpec("boston", 531, "regression", 506, "classic non-linear housing (known ethical caveat)"),
    DatasetSpec("bodyfat", 560, "regression", 252, "non-linear body measurements"),
    DatasetSpec("no2", 547, "regression", 500, "air-quality, non-linear"),
    DatasetSpec("pm10", 522, "regression", 500, "air-quality, non-linear"),
    DatasetSpec("sensory", 546, "regression", 576, "wine sensory scores"),
    DatasetSpec("cloud", 210, "regression", 108, "small non-linear"),
    DatasetSpec("autoPrice", 207, "regression", 159, "non-linear car pricing"),
    DatasetSpec("stock", 223, "regression", 950, "non-linear financial"),
]

BENCHMARK_DATASETS: list[DatasetSpec] = CLASSIFICATION_DATASETS + REGRESSION_DATASETS


@dataclass
class LoadedDataset:
    """A dataset loaded into model-ready arrays."""

    X: np.ndarray
    y: np.ndarray
    task: Task
    name: str
    data_id: int
    feature_names: list[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int | None:
        return int(len(np.unique(self.y))) if self.task == "classification" else None


class OpenMLBenchmark:
    """Load and evaluate the curated suite of small, non-linear OpenML datasets."""

    def __init__(self, task: Task | None = None, cache_dir: str | None = None) -> None:
        """Args:
        task: optionally restrict the suite to ``"classification"`` or ``"regression"``.
        cache_dir: ``data_home`` passed to ``fetch_openml`` (defaults to scikit-learn's cache).
        """
        if task is not None and task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification', 'regression' or None, got {task!r}")
        self.task = task
        self.cache_dir = cache_dir
        self.specs: list[DatasetSpec] = [
            s for s in BENCHMARK_DATASETS if task is None or s.task == task
        ]
        self._by_key: dict[str | int, DatasetSpec] = {}
        for s in self.specs:
            self._by_key[s.name] = s
            self._by_key[s.data_id] = s

    def list(self) -> pd.DataFrame:
        """Return the registry as a DataFrame (name, data_id, task, n_rows, note)."""
        return pd.DataFrame(
            [(s.name, s.data_id, s.task, s.n_rows, s.note) for s in self.specs],
            columns=["name", "data_id", "task", "n_rows", "note"],
        )

    def _spec(self, name_or_id: str | int | DatasetSpec) -> DatasetSpec:
        if isinstance(name_or_id, DatasetSpec):
            return name_or_id
        try:
            return self._by_key[name_or_id]
        except KeyError as exc:
            raise KeyError(
                f"{name_or_id!r} is not in this suite. Available: "
                f"{[s.name for s in self.specs]}"
            ) from exc

    def load(self, name_or_id: str | int | DatasetSpec) -> LoadedDataset:
        """Load a dataset by name or data_id into model-ready ``np.float32`` arrays.

        Categorical features are one-hot encoded and missing values imputed; the classification target
        is label-encoded to integers and the regression target cast to float.
        """
        spec = self._spec(name_or_id)
        bunch = fetch_openml(
            data_id=spec.data_id, as_frame=True, parser="auto", data_home=self.cache_dir
        )
        X_df: pd.DataFrame = bunch.data
        y_raw: pd.Series = bunch.target

        if spec.drop_cols:
            X_df = X_df.drop(columns=list(spec.drop_cols), errors="ignore")

        # Anything non-numeric (object / category / pandas string / bool) is one-hot encoded; the rest
        # is numeric. Booleans are forced categorical so they are encoded rather than median-imputed.
        categorical = [
            c
            for c in X_df.columns
            if is_bool_dtype(X_df[c]) or not is_numeric_dtype(X_df[c])
        ]
        numeric = [c for c in X_df.columns if c not in categorical]

        transformers = []
        if numeric:
            transformers.append(
                ("num", SimpleImputer(strategy="median"), numeric)
            )
        if categorical:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical,
                )
            )
        pre = ColumnTransformer(transformers, remainder="drop")
        X = pre.fit_transform(X_df).astype(np.float32)
        feature_names = list(pre.get_feature_names_out())

        if spec.task == "classification":
            y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(np.int64)
        else:
            y = np.asarray(y_raw, dtype=np.float32)

        return LoadedDataset(
            X=X,
            y=y,
            task=spec.task,
            name=spec.name,
            data_id=spec.data_id,
            feature_names=feature_names,
        )

    def splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: Task,
        n_splits: int = 5,
        train_size: float | int = 0.6,
    ):
        """Yield ``(train_idx, test_idx)`` pairs.

        Stratified for classification (``StratifiedShuffleSplit``) and plain ``ShuffleSplit`` for
        regression, matching the project's existing convention with ``random_state=0``.
        """
        splitter_cls = StratifiedShuffleSplit if task == "classification" else ShuffleSplit
        splitter = splitter_cls(
            n_splits=n_splits, train_size=train_size, random_state=RANDOM_STATE
        )
        yield from splitter.split(X, y)

    def evaluate(
        self,
        estimator_factory: Callable[[Task], object],
        datasets: list[str] | None = None,
        task: Task | None = None,
        n_splits: int = 5,
        train_size: float | int = 0.6,
        scale: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Evaluate an estimator across datasets with repeated splits.

        Args:
            estimator_factory: callable ``task -> fresh estimator`` (e.g. ``lambda t: TabPFNClassifier()``
                for classification, or returning a student model). A fresh estimator is built per split.
            datasets: subset of dataset names to run (defaults to the whole suite / current task filter).
            task: optionally restrict to one task (in addition to any filter set in ``__init__``).
            n_splits: number of repeated shuffle splits.
            train_size: fraction or absolute number of training rows (rest is the test set).
            scale: standardize features with ``StandardScaler`` (helps student models; harmless to TabPFN).
            verbose: print per-dataset results as they complete.

        Returns:
            DataFrame with one row per dataset. Classification reports ``acc_mean``/``acc_std``;
            regression reports ``r2_mean``/``r2_std`` and ``rmse_mean``/``rmse_std``.
        """
        specs = self.specs
        if task is not None:
            specs = [s for s in specs if s.task == task]
        if datasets is not None:
            wanted = set(datasets)
            specs = [s for s in specs if s.name in wanted]
        if not specs:
            raise ValueError("No datasets selected. Check the `datasets`/`task` filters.")

        rows = []
        for spec in specs:
            ds = self.load(spec)
            primary, secondary = [], []  # accuracy, or (r2, rmse)
            for train_idx, test_idx in self.splits(
                ds.X, ds.y, ds.task, n_splits=n_splits, train_size=train_size
            ):
                X_tr, X_te = ds.X[train_idx], ds.X[test_idx]
                y_tr, y_te = ds.y[train_idx], ds.y[test_idx]
                if scale:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
                    X_te = scaler.transform(X_te).astype(np.float32)

                model = estimator_factory(ds.task)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)

                if ds.task == "classification":
                    classes = np.unique(ds.y)
                    primary.append(accuracy_score(y_te, _to_class_labels(pred, classes)))
                    secondary.append(_roc_auc(y_te, _class_scores(model, X_te), classes))
                else:
                    primary.append(r2_score(y_te, pred))
                    secondary.append(np.sqrt(mean_squared_error(y_te, pred)))

            if spec.task == "classification":
                row = {
                    "name": spec.name,
                    "task": spec.task,
                    "n_rows": spec.n_rows,
                    "acc_mean": float(np.mean(primary)),
                    "acc_std": float(np.std(primary)),
                    "auc_mean": float(np.nanmean(secondary)),
                    "auc_std": float(np.nanstd(secondary)),
                }
            else:
                row = {
                    "name": spec.name,
                    "task": spec.task,
                    "n_rows": spec.n_rows,
                    "r2_mean": float(np.mean(primary)),
                    "r2_std": float(np.std(primary)),
                    "rmse_mean": float(np.mean(secondary)),
                    "rmse_std": float(np.std(secondary)),
                }
            rows.append(row)
            if verbose:
                metric = (
                    f"acc={row['acc_mean']:.3f}+/-{row['acc_std']:.3f} "
                    f"auc={row['auc_mean']:.3f}+/-{row['auc_std']:.3f}"
                    if spec.task == "classification"
                    else f"r2={row['r2_mean']:.3f}+/-{row['r2_std']:.3f}"
                )
                print(f"[{spec.task:14s}] {spec.name:20s} {metric}")

        return pd.DataFrame(rows)


def main() -> None:
    """Light demo: print the registry and load one classification dataset."""
    bench = OpenMLBenchmark()
    print("Benchmark suite (61 datasets):")
    print(bench.list().to_string(index=False))

    ds = bench.load(CLASSIFICATION_DATASETS[0].name)
    print(f"\nLoaded {ds.name!r}: X={ds.X.shape} ({ds.X.dtype}), y={ds.y.shape}")
    print(f"classes: {ds.n_classes}, feature count: {len(ds.feature_names)}")


if __name__ == "__main__":
    main()
