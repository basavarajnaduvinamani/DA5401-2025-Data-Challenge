# -*- coding: utf-8 -*-


import os
import sys
import json
import time
import random
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


class ProjectConfig:
    PROJECT_NAME = "Deep_Metric_Learning_Evaluator"
    VERSION = "3.0.0"
    AUTHOR = "DA25C005"

    SEED = 42

    BASE_PATH = "/kaggle/input/da5401-2025-data-challenge"
    OUTPUT_PATH = "/kaggle/working"

    TRAIN_DATA = os.path.join(BASE_PATH, "train_data.json")
    TEST_DATA = os.path.join(BASE_PATH, "test_data.json")
    METRIC_META = os.path.join(BASE_PATH, "metric_names.json")
    METRIC_EMBED = os.path.join(BASE_PATH, "metric_name_embeddings.npy")
    SUBMISSION_FILE = os.path.join(OUTPUT_PATH, "submission.csv")

    TEXT_MAX_FEATURES = 50000
    TEXT_NGRAM_RANGE = (1, 3)
    SVD_COMPONENTS = 100
    SVD_ITERATIONS = 10

    MLP_HIDDEN_LAYERS = [512, 128]
    MLP_DROPOUT = 0.3
    MLP_ACTIVATION = "ReLU"

    RIDGE_ALPHA = 1.0

    SVR_C = 1.0
    SVR_EPSILON = 0.1
    SVR_KERNEL = 'rbf'

    N_FOLDS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 25
    PATIENCE = 5

    WEIGHT_MLP = 0.50
    WEIGHT_RIDGE = 0.30
    WEIGHT_SVR = 0.20

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2
    PIN_MEMORY = True


class SystemUtils:
    @staticmethod
    def seed_everything(seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def get_logger(name: str = "Training"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

logger = SystemUtils.get_logger()



class DataLoaderService:
    def __init__(self, config: ProjectConfig):
        self.cfg = config
        self.metric_map = {}
        self.metric_embeddings = None

    def load_metadata(self):
        logger.info(f"Loading metric metadata from {self.cfg.METRIC_META}")
        with open(self.cfg.METRIC_META, 'r') as f:
            names = json.load(f)
        self.metric_map = {name: idx for idx, name in enumerate(names)}

        logger.info(f"Loading metric embeddings from {self.cfg.METRIC_EMBED}")
        self.metric_embeddings = np.load(self.cfg.METRIC_EMBED)

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Loading training and testing datasets...")
        train_df = pd.read_json(self.cfg.TRAIN_DATA)
        test_df = pd.read_json(self.cfg.TEST_DATA)
        logger.info(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")
        return train_df, test_df

class TextProcessor:
    def __init__(self):
        pass

    def construct_rich_input(self, df: pd.DataFrame) -> pd.Series:
        cols = ["prompt", "response", "system_prompt", "expected_response"]
        valid = [c for c in cols if c in df.columns]

        logger.info(f"Constructing rich text from fields: {valid}")
        base_text = df[valid[0]].fillna("").astype(str)

        for col in valid[1:]:
            base_text = base_text + " [SEP] " + df[col].fillna("").astype(str)

        return base_text


class LatentSemanticAnalyzer:
    def __init__(self, config: ProjectConfig):
        self.cfg = config
        self.tfidf = TfidfVectorizer(
            max_features=self.cfg.TEXT_MAX_FEATURES,
            ngram_range=self.cfg.TEXT_NGRAM_RANGE,
            min_df=3,
            strip_accents='unicode',
            sublinear_tf=True
        )
        self.svd = TruncatedSVD(
            n_components=self.cfg.SVD_COMPONENTS,
            n_iter=self.cfg.SVD_ITERATIONS,
            random_state=self.cfg.SEED
        )

    def fit_transform(self, text: pd.Series) -> np.ndarray:
        logger.info("Generating TF-IDF matrix...")
        tfidf_matrix = self.tfidf.fit_transform(text)
        logger.info(f"TF-IDF Shape: {tfidf_matrix.shape}")

        logger.info("Applying Truncated SVD decomposition...")
        svd_matrix = self.svd.fit_transform(tfidf_matrix)
        logger.info(f"LSA Features Shape: {svd_matrix.shape}")
        return svd_matrix

    def transform(self, text: pd.Series) -> np.ndarray:
        tfidf_matrix = self.tfidf.transform(text)
        return self.svd.transform(tfidf_matrix)

class FeatureFusionEngine:
    def __init__(self, config: ProjectConfig):
        self.cfg = config
        self.emb_scaler = StandardScaler()
        self.aux_scaler = MinMaxScaler()

    def process_metric_embeddings(self, df: pd.DataFrame,
                                embed_matrix: np.ndarray,
                                mapping: Dict,
                                fit: bool = False) -> np.ndarray:
        indices = [mapping.get(n, -1) for n in df["metric_name"]]
        safe_indices = [i if i != -1 else 0 for i in indices]

        raw_embs = embed_matrix[safe_indices]
        mask = np.array([1 if i != -1 else 0 for i in indices]).reshape(-1, 1)
        masked_embs = raw_embs * mask

        if fit:
            return self.emb_scaler.fit_transform(masked_embs)
        return self.emb_scaler.transform(masked_embs)

    def process_auxiliary(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        p_len = df['prompt'].fillna("").str.len()
        r_len = df['response'].fillna("").str.len() if 'response' in df.columns else np.zeros(len(df))

        features = np.column_stack([
            np.log1p(p_len),
            np.log1p(r_len),
            p_len / (r_len + 1.0)
        ])

        if fit:
            return self.aux_scaler.fit_transform(features)
        return self.aux_scaler.transform(features)

    def fuse(self, svd_feats: np.ndarray, metric_feats: np.ndarray, aux_feats: np.ndarray) -> np.ndarray:
        return np.hstack([svd_feats, metric_feats, aux_feats])


class AbstractModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim: int, layers: List[int], dropout: float):
        super(PyTorchMLP, self).__init__()

        net_layers = []
        in_d = input_dim

        for out_d in layers:
            net_layers.append(nn.Linear(in_d, out_d))
            net_layers.append(nn.BatchNorm1d(out_d))
            net_layers.append(nn.ReLU())
            net_layers.append(nn.Dropout(dropout))
            in_d = out_d

        net_layers.append(nn.Linear(in_d, 1))
        self.model = nn.Sequential(*net_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x).squeeze(1)

class FitnessDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class MLPWrapper(AbstractModel):
    def __init__(self, config: ProjectConfig, input_dim: int):
        self.cfg = config
        self.input_dim = input_dim
        self.model = PyTorchMLP(input_dim, config.MLP_HIDDEN_LAYERS, config.MLP_DROPOUT).to(config.DEVICE)
        self.optimizer = AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=2, factor=0.5)

    def fit(self, X, y, X_val, y_val):
        train_ds = FitnessDataset(X, y)
        val_ds = FitnessDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=self.cfg.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.BATCH_SIZE)

        best_loss = float('inf')
        best_state = None

        for epoch in range(self.cfg.EPOCHS):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.cfg.DEVICE), by.to(self.cfg.DEVICE)
                self.optimizer.zero_grad()
                out = self.model(bx)
                loss = self.criterion(out, by)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.cfg.DEVICE), by.to(self.cfg.DEVICE)
                    out = self.model(bx)
                    val_loss += self.criterion(out, by).item()

            avg_val_loss = np.sqrt(val_loss / len(val_loader))
            self.scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_state = self.model.state_dict()

        self.model.load_state_dict(best_state)
        return best_loss

    def predict(self, X):
        ds = FitnessDataset(X)
        loader = DataLoader(ds, batch_size=self.cfg.BATCH_SIZE)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for bx in loader:
                bx = bx.to(self.cfg.DEVICE)
                out = self.model(bx)
                preds.extend(out.cpu().numpy())
        return np.array(preds)

class RidgeWrapper(AbstractModel):
    def __init__(self, config: ProjectConfig):
        self.model = Ridge(alpha=config.RIDGE_ALPHA, random_state=config.SEED)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class SVRWrapper(AbstractModel):
    def __init__(self, config: ProjectConfig):
        self.model = SVR(C=config.SVR_C, epsilon=config.SVR_EPSILON, kernel=config.SVR_KERNEL)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class EnsembleOrchestrator:
    def __init__(self, config: ProjectConfig):
        self.cfg = config
        self.kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    def run(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        oof_mlp = np.zeros(len(X))
        oof_ridge = np.zeros(len(X))
        oof_svr = np.zeros(len(X))

        test_mlp = np.zeros(len(X_test))
        test_ridge = np.zeros(len(X_test))
        test_svr = np.zeros(len(X_test))

        logger.info(f"Starting {self.cfg.N_FOLDS}-Fold Cross Validation...")

        for fold, (train_idx, val_idx) in enumerate(self.kf.split(X)):
            print(f"\n>>> Processing Fold {fold + 1} <<<")

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Train MLP
            mlp = MLPWrapper(self.cfg, input_dim=X.shape[1])
            loss = mlp.fit(X_tr, y_tr, X_val, y_val)
            oof_mlp[val_idx] = mlp.predict(X_val)
            test_mlp += mlp.predict(X_test) / self.cfg.N_FOLDS
            print(f"    MLP RMSE: {loss:.4f}")

            # Train Ridge
            ridge = RidgeWrapper(self.cfg)
            ridge.fit(X_tr, y_tr)
            oof_ridge[val_idx] = ridge.predict(X_val)
            test_ridge += ridge.predict(X_test) / self.cfg.N_FOLDS
            rmse_r = mean_squared_error(y_val, oof_ridge[val_idx], squared=False)
            print(f"    Ridge RMSE: {rmse_r:.4f}")

            # Train SVR
            svr = SVRWrapper(self.cfg)
            svr.fit(X_tr, y_tr)
            oof_svr[val_idx] = svr.predict(X_val)
            test_svr += svr.predict(X_test) / self.cfg.N_FOLDS
            rmse_s = mean_squared_error(y_val, oof_svr[val_idx], squared=False)
            print(f"    SVR RMSE: {rmse_s:.4f}")

        # Ensemble Aggregation
        final_oof = (
            self.cfg.WEIGHT_MLP * oof_mlp +
            self.cfg.WEIGHT_RIDGE * oof_ridge +
            self.cfg.WEIGHT_SVR * oof_svr
        )

        final_rmse = mean_squared_error(y, final_oof, squared=False)
        logger.info(f"Final Ensemble OOF RMSE: {final_rmse:.4f}")

        final_test_preds = (
            self.cfg.WEIGHT_MLP * test_mlp +
            self.cfg.WEIGHT_RIDGE * test_ridge +
            self.cfg.WEIGHT_SVR * test_svr
        )

        return final_test_preds


def main():
    # Initialize Environment
    SystemUtils.seed_everything(ProjectConfig.SEED)
    logger.info(f"Project: {ProjectConfig.PROJECT_NAME} | Ver: {ProjectConfig.VERSION}")

    # Load Data
    loader = DataLoaderService(ProjectConfig)
    loader.load_metadata()
    train_df, test_df = loader.load_datasets()

    # Preprocessing
    processor = TextProcessor()
    train_text = processor.construct_rich_text(train_df)
    test_text = processor.construct_rich_text(test_df)

    # Feature Engineering
    lsa = LatentSemanticAnalyzer(ProjectConfig)
    fusion = FeatureFusionEngine(ProjectConfig)

    X_lsa_train = lsa.fit_transform(train_text)
    X_lsa_test = lsa.transform(test_text)

    X_met_train = fusion.process_metric_embeddings(train_df, loader.metric_embeddings, loader.metric_map, fit=True)
    X_met_test = fusion.process_metric_embeddings(test_df, loader.metric_embeddings, loader.metric_map)

    X_aux_train = fusion.process_auxiliary(train_df, fit=True)
    X_aux_test = fusion.process_auxiliary(test_df)

    X_train = fusion.fuse(X_lsa_train, X_met_train, X_aux_train)
    X_test = fusion.fuse(X_lsa_test, X_met_test, X_aux_test)
    y_train = train_df['score'].values

    # Modeling
    orchestrator = EnsembleOrchestrator(ProjectConfig)
    predictions = orchestrator.run(X_train, y_train, X_test)

    # Submission Generation
    final_preds = np.clip(predictions, 0, 10)

    if 'id' in test_df.columns:
        ids = test_df['id']
    else:
        ids = np.arange(1, len(test_df) + 1)

    sub = pd.DataFrame({"ID": ids, "score": final_preds})

    if not os.path.exists(ProjectConfig.OUTPUT_DIR):
        os.makedirs(ProjectConfig.OUTPUT_DIR)

    sub.to_csv(ProjectConfig.SUBMISSION_FILE, index=False)
    logger.info(f"Successfully saved submission to {ProjectConfig.SUBMISSION_FILE}")

if __name__ == "__main__":
    main()