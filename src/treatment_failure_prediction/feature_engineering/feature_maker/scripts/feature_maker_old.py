import pandas as pd
import polars as pl
from tqdm import tqdm
from timeseriesflattener import (
    StaticFrame,
    ValueFrame,
    TimestampValueFrame,
    PredictionTimeFrame,
    OutcomeSpec,
    BooleanOutcomeSpec,
    StaticSpec,
    PredictorSpec,
    Flattener,
    CountAggregator,
)

from multiprocessing import Pool
import numpy as np


class FeatureMaker:
    def __init__(
        self,
        long_data: pd.DataFrame,
        wide_data: pd.DataFrame,
        # test_patientids: pd.Series,
        entity_id_col_name: str = "patientid",
        timestamp_col_name: str = "timestamp",
    ) -> None:
        """Class for creating a feature matrix from health data.

        Args:
            long_data (pd.DataFrame): Long data with columns entity_id_col, timestamp_col, variable_code, value and data_source.
            wide_data (pd.DataFrame): Wide data containing entity_id_col.
            entity_id_col_name (str, optional): String specifying the name of the entity_id_col. Defaults to "patientid".
            timestamp_col_name (str, optional): String specifying the name of the timestamp_col. Defaults to "timestamp".
        """
        self.long_data = long_data
        self.wide_data = wide_data
        self.entity_id_col_name = entity_id_col_name
        self.timestamp_col_name = timestamp_col_name
        # self.test_patientids = test_patientids

        self.make_categorical_features_integers()

        self.predictiontime_frame = None
        self.specifications = []
        self.static_specifications = []
        self.dynamic_specifications = []
        self.dynamic_features = []
        self.static_features = []
        self.outcome_features = []
        self.features = []

    def _reset_all_features(self):
        """Resets all specifications for the class"""
        self.predictiontime_frame = None
        self.specifications = []
        self.static_specifications = []
        self.dynamic_specifications = []
        self.dynamic_features = []
        self.static_features = []
        self.outcome_features = []
        self.features = []

    def add_dynamic_feature(self, specification: dict) -> None:
        """Add a dynamic feature to the feature matrix

        Args:
            specification (dict): Dictionary containing the specification of how the feature should be calculated
        """
        self.dynamic_specifications.append(specification)

    def add_static_feature(self, specification: dict) -> None:
        """Add a static feature to the feature matrix

        Args:
            specification (dict): Dictionary containing the specification of how the feature should be calculated
        """
        self.static_specifications.append(specification)

    def _filter_long_data_by_data_source(self, data_source: str) -> pd.DataFrame:
        """Helper function for filtering the long format data by data source

        Args:
            data_source (str): Name of the data source to filter by

        Returns:
            pd.DataFrame: Dataframe containing only the specified data source
        """
        return self.long_data[self.long_data["data_source"] == data_source].reset_index(
            drop=True
        )

    def make_categorical_features_integers(self):
        """Converts all categorical features into dummy coded integer values"""
        column_types = self.wide_data.dtypes.reset_index()
        columns_to_convert = column_types[
            (column_types[0] == "object")
            & (column_types["index"] != "prediction_time_uuid")
        ]["index"].values
        for column in columns_to_convert:
            self.wide_data[column] = pd.Categorical(self.wide_data[column]).codes

    def _filter_and_rename_data(
        self, filtered_data: pd.DataFrame, specification: dict
    ) -> pd.DataFrame:
        """Helper function for filtering and renaming the long format data

        Args:
            specification (dict): Specification dictionary containing info about the feature

        Returns:
            pd.DataFrame: Dataframe ready for feature extraction
        """
        column_name = (
            f"{specification.get('data_source')}_{specification.get('variable_value')}"
        )

        data = filtered_data[
            filtered_data[specification.get("variable_column")]
            == specification.get("variable_value")
        ].reset_index(drop=True)

        data = data.rename(
            columns={
                specification.get("value_column"): column_name,
            }
        )[
            [self.timestamp_col_name, self.entity_id_col_name, column_name]
        ].reset_index(drop=True)

        # NO SCALING

        if not specification.get("categorical"):
            # NOTE: Scaling should be done per each split, so should be put in Pipeline
            data[column_name] = pd.to_numeric(data[column_name], errors="coerce")
        #     # scale currently only min max - consider using standard scaler
        #     min_value, max_value = (
        #         data[column_name].min(),
        #         data[column_name].max(),
        #     )
        #     data[column_name] = (data[column_name] - min_value) / (
        #         max_value - min_value
        #     )

        return ValueFrame(data, entity_id_col_name=self.entity_id_col_name)

    def specify_prediction_time_from_wide_format(self, date_column: str):
        """Specify the prediction times for each patient from the wide format

        Args:
            date_column (str): Name of the column containing the prediction time for each patient
        """
        self.predictiontime_frame = PredictionTimeFrame(
            self.wide_data[[self.entity_id_col_name, date_column]]
            .rename(columns={date_column: self.timestamp_col_name})
            .reset_index(drop=True),
            entity_id_col_name=self.entity_id_col_name,
            timestamp_col_name=self.timestamp_col_name,
        )

    def add_features_given_ratio(
        self,
        data_source: str,
        agg_funcs: list,
        lookbacks: list,
        fallback: int = -1,
        categorical: bool = False,
        variable_col: str = "variable_code",
        value_column: str = "value",
        proportion: int = 0.2,
        collapse_rare_conditions_to_feature: bool = False,
    ):
        """Adds all dynamic features from a datasource for variables present in more than a specified proportion of patients.
        Note that this proportion is being calculated from the entire period of the data i.e. also includes data from after the prediction time.

        Args:
            data_source (str): Name of the data source
            agg_funcs (list): Aggregation functions from timeseriesflattener to be used for making features
            lookbacks (list): Lookbacks used for aggregation
            fallback (int, optional): Fallback if no value can be found. Defaults to -1.
            categorical (bool, optional): Boolean value indicating whether values are categorical. Defaults to False.
            variable_col (str, optional): Variable column name. Defaults to "variable_code".
            proportion (int, optional): Proportion of patients where the variable is present. Defaults to 0.2.
            collapse_rare_conditions_to_feature (bool, optional): Boolean indicating whether to count rare occurences as a feature. Defaults to False.
        """
        subset = self.long_data[
            self.long_data["data_source"] == data_source
        ].reset_index(drop=True)

        # ensure that feature generation is done only on the basis of training
        # subset = subset[
        #     ~subset[self.entity_id_col_name].isin(self.test_patientids)
        # ].reset_index(drop=True)

        subset_codes = (
            subset.groupby([self.entity_id_col_name, variable_col])
            .count()
            .reset_index()
        )
        values = subset_codes[variable_col].value_counts().reset_index()
        n_values = len(values)
        values[variable_col] = values[variable_col] / len(self.wide_data)
        values = values[values[variable_col] > proportion]
        values = values["index"].values

        if collapse_rare_conditions_to_feature and n_values != len(values):
            self.long_data.loc[
                (self.long_data["data_source"] == data_source)
                & (~self.long_data[variable_col].isin(values)),
                variable_col,
            ] = "rare"
            specification = {
                "data_source": f"{data_source}",
                "base_name": f"{data_source}_rare",
                "lookbacks": lookbacks,
                "variable_column": variable_col,
                "value_column": value_column,
                "variable_value": "rare",
                "agg_function": [
                    CountAggregator()
                ],  # consider whether this makes anysense - shouldn't this always just be counts? can't really do anything else
                "fallback": fallback,
                "categorical": categorical,
            }
            self.add_dynamic_feature(specification=specification)

        for value in values:
            specification = {
                "data_source": f"{data_source}",
                "base_name": f"{data_source}_{value}",
                "lookbacks": lookbacks,
                "variable_column": variable_col,
                "value_column": value_column,
                "variable_value": value,
                "agg_function": agg_funcs,
                "fallback": fallback,
                "categorical": categorical,
            }
            self.add_dynamic_feature(specification=specification)

    def filter_long_format_from_wide_format_ids(self):
        """Using the wide format patient_ids, filter the long format"""
        self.long_data = self.long_data[
            self.long_data[self.entity_id_col_name].isin(
                self.wide_data[self.entity_id_col_name]
            )
        ].reset_index(drop=True)

    def add_outcome_from_wide_format(
        self,
        date_column: str,
        outcome_column: str,
        lookahead_distances: tuple,
        aggregators: tuple,
        fallback: int = 0,
    ) -> None:
        """Adds an outcome from the wide format data.

        Args:
            date_column (str): Name of the column containing the date when outcome happened
            outcome_column (str): Name of the column containing the label of the outcome
            lookahead_distances (tuple): Iterable of different timedeltas to use as lookahead distances
            aggregators (tuple): Iterable of different aggregators to use for outcome definition.
            fallback (int, optional): Fallback value in case of NA. Defaults to 0.
        """
        self.outcome_features.append(
            OutcomeSpec(
                value_frame=ValueFrame(
                    init_df=self.wide_data[
                        [self.entity_id_col_name, date_column, outcome_column]
                    ]
                    .rename(
                        columns={
                            date_column: self.timestamp_col_name,
                        },
                    )
                    .reset_index(drop=True),
                    entity_id_col_name=self.entity_id_col_name,
                ),
                lookahead_distances=lookahead_distances,
                aggregators=aggregators,
                fallback=fallback,
            )
        )

    def add_outcome_from_long_format(
        self,
        data_source: str,
        outcome_name: str,
        lookahead_distances: tuple,
        aggregators: tuple,
        fallback: int = 0,
    ) -> None:
        """Adds an outcome from the wide format data.

        Args:
            date_column (str): Name of the column containing the date when outcome happened
            outcome_column (str): Name of the column containing the label of the outcome
            lookahead_distances (tuple): Iterable of different timedeltas to use as lookahead distances
            aggregators (tuple): Iterable of different aggregators to use for outcome definition.
            fallback (int, optional): Fallback value in case of NA. Defaults to 0.
        """

        filtered_long_data = (
            self.long_data[self.long_data["data_source"] == data_source]
            .rename(columns={"value": outcome_name})
            .reset_index(drop=True)[
                [self.entity_id_col_name, self.timestamp_col_name, outcome_name]
            ]
        )
        self.outcome_features.append(
            OutcomeSpec(
                value_frame=ValueFrame(
                    init_df=filtered_long_data,
                    entity_id_col_name=self.entity_id_col_name,
                ),
                lookahead_distances=lookahead_distances,
                aggregators=aggregators,
                fallback=fallback,
            )
        )

    def _make_dynamic_feature_from_specification(
        self,
        filtered_data: pd.DataFrame,
        specification: dict,
    ) -> None:
        """Helper function for creating a dynamic feature given a specification.

        Args:
            filtered_data (pd.DataFrame): Filtered data containing only the values of one data source
            specification (dict): Specification of the dynamic feature
        """

        value_frame = self._filter_and_rename_data(filtered_data, specification)

        feature = PredictorSpec(
            value_frame=value_frame,
            lookbehind_distances=specification.get("lookbacks"),
            fallback=specification.get("fallback"),
            aggregators=specification.get("agg_function"),
            # feature_base_name=specification.get("base_name"),
        )
        self.dynamic_features.append(feature)

    def _make_static_feature_from_specification(self, specification: dict) -> None:
        """Helper function for creating a static feature from specification

        Args:
            specification (dict): Dictionary of specification
        """

        selected_data = StaticFrame(
            self.wide_data[
                [self.entity_id_col_name, specification.get("value_column")]
            ].reset_index(drop=True),
            entity_id_col_name=self.entity_id_col_name,
        )

        self.static_features.append(
            StaticSpec(
                value_frame=selected_data,
                fallback=-1,  # NOTE: HARDCODED AS -1
                column_prefix="pred_RKKP",
            )
        )

    def _make_all_static_features(self) -> None:
        """Helper function for creating all static features specified in the class"""
        print("Creating static features:")
        for specification in tqdm(self.static_specifications):
            self._make_static_feature_from_specification(specification)

    def _make_all_dynamic_features(self) -> None:
        """Helper function for creating all dynamic features specified in the class"""
        print("Creating dynamic features:")
        list_of_all_data_sources = set(
            [
                specification.get("data_source")
                for specification in self.dynamic_specifications
            ]
        )
        # from functools import partial

        for data_source in list_of_all_data_sources:
            print(f"Creating features for {data_source}:")
            filtered_data = self._filter_long_data_by_data_source(
                data_source=data_source
            )
            relevant_specifications = [
                x
                for x in self.dynamic_specifications
                if x.get("data_source") == data_source
            ]

            # NOTE: attempt at making feature generation parallel. "processes should be specificed"
            # with Pool(processes=processes) as pool:  # need to make flexible
            #     for _ in tqdm(
            #         pool.imap_unordered(
            #             partial(
            #                 self._make_dynamic_feature_from_specification,
            #                 filtered_data,
            #             ),
            #             relevant_specifications,
            #         ),
            #         total=len(relevant_specifications),
            #     ):
            #         pass

            for specification in tqdm(relevant_specifications):
                self._make_dynamic_feature_from_specification(
                    filtered_data,
                    specification,
                )

    def make_all_features(self) -> None:
        """Create all features specified"""
        if self.static_specifications:
            self._make_all_static_features()
        if self.dynamic_specifications:
            self._make_all_dynamic_features()
        temp_features = self.dynamic_features.copy()
        temp_features.extend(self.static_features)
        temp_features.extend(self.outcome_features)
        self.features = temp_features

    def add_feature_from_polars_dataframe(
        self,
        dataframe: pl.DataFrame,
        aggregators: list,
        lookbehind_distances: list,
        column_prefix: str,
        fallback=-1,
    ) -> None:
        spec = PredictorSpec(
            ValueFrame(
                init_df=dataframe,
                entity_id_col_name=self.entity_id_col_name,
                value_timestamp_col_name=self.timestamp_col_name,
            ),
            lookbehind_distances=lookbehind_distances,
            aggregators=aggregators,
            fallback=fallback,
            column_prefix=column_prefix,
        )
        self.features.append(spec)

    def create_feature_matrix(self, n_workers: int) -> pd.DataFrame:
        """Create the feature matrix from all the specifications

        Args:
            n_workers (int): n_workers for parallel processing

        Returns:
            pd.DataFrame: Feature matrix dataframe
        """
        if not self.features:
            self.make_all_features()

        self.flattener = Flattener(
            predictiontime_frame=self.predictiontime_frame,
            # compute_lazily=True,
            n_workers=n_workers,
        )

        feature_matrix = self.flattener.aggregate_timeseries(
            specs=self.features
        ).collect()
        # NOTE: This is not a sustainable fix - outcome labels are also changed to -1

        # feature_matrix = feature_matrix.with_columns(pl.all().fill_null(-1))
        feature_matrix = feature_matrix.to_pandas()
        self.feature_matrix = feature_matrix

        # fallback doesn't seem to work, so now we're just doing it ourselves.
        return self.feature_matrix
