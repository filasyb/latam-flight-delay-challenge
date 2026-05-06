import pandas as pd

from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression

class DelayModel:
    TARGET_COLUMN = "delay"
    DELAY_THRESHOLD_IN_MINUTES = 15
    FEATURE_SOURCE_COLUMNS = ["OPERA", "MES", "TIPOVUELO"]
    TARGET_SOURCE_COLUMNS = ["Fecha-I", "Fecha-O"]
    FEATURE_COLUMNS = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    def __init__(
        self
    ):
        self._model = None

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        self._validate_columns(data, self.FEATURE_SOURCE_COLUMNS)
        data = data.copy()

        if target_column == self.TARGET_COLUMN and target_column not in data.columns:
            self._validate_columns(data, self.TARGET_SOURCE_COLUMNS)
            data[target_column] = self._get_delay(data)

        features = pd.get_dummies(
            data[self.FEATURE_SOURCE_COLUMNS],
            columns=self.FEATURE_SOURCE_COLUMNS
        )
        features = features.reindex(columns=self.FEATURE_COLUMNS, fill_value=0)

        if target_column:
            self._validate_columns(data, [target_column])
            target = data[[target_column]]
            return features, target

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        target_series = target.squeeze()
        n_negative = (target_series == 0).sum()
        n_positive = (target_series == 1).sum()
        total = len(target_series)

        self._model = LogisticRegression(
            class_weight={
                0: n_positive / total,
                1: n_negative / total,
            },
            max_iter=1000,
        )
        self._model.fit(features, target_series)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            return [0 for _ in range(features.shape[0])]

        return [int(prediction) for prediction in self._model.predict(features)]

    def _get_delay(
        self,
        data: pd.DataFrame
    ) -> List[int]:
        scheduled_dates = pd.to_datetime(data["Fecha-I"])
        operation_dates = pd.to_datetime(data["Fecha-O"])
        min_diff = (operation_dates - scheduled_dates).dt.total_seconds() / 60

        return [int(delay) for delay in min_diff > self.DELAY_THRESHOLD_IN_MINUTES]

    def _validate_columns(
        self,
        data: pd.DataFrame,
        required_columns: List[str]
    ) -> None:
        missing_columns = [
            column for column in required_columns if column not in data.columns
        ]

        if missing_columns:
            missing_columns_text = ", ".join(missing_columns)
            raise ValueError(f"Missing required columns: {missing_columns_text}")
