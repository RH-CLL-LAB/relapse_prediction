from timeseriesflattener.aggregators import (
    MaxAggregator,
    MinAggregator,
    MeanAggregator,
    CountAggregator,
    SumAggregator,
    VarianceAggregator,
    HasValuesAggregator,
    SlopeAggregator,
    LatestAggregator,
    EarliestAggregator,
)

feature_specs = [
    {
        "data_source": "ordered_medicine",
        "agg_funcs": [
            CountAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "labmeasurements",
        "agg_funcs": [
            CountAggregator(),
            MaxAggregator(),
            SlopeAggregator(timestamp_col_name="timestamp"),
            MinAggregator(),
            LatestAggregator(timestamp_col_name="timestamp"),
            VarianceAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "lab_measurements_data_all",
        "agg_funcs": [
            CountAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "sks_referals",
        "agg_funcs": [CountAggregator(), SumAggregator()],
        "proportion": 0.1,
    },
    {
        "data_source": "sks_at_the_hospital",
        "agg_funcs": [CountAggregator(), SumAggregator()],
        "proportion": 0.1,
    },
    {
        "data_source": "sks_referals_unique",
        "agg_funcs": [CountAggregator()],
        "proportion": 0.1,
    },
    {
        "data_source": "sks_at_the_hospital_unique",
        "agg_funcs": [CountAggregator()],
        "proportion": 0.1,
    },
    {"data_source": "SDS_pato", "agg_funcs": [CountAggregator()], "proportion": 0.1},
    {
        "data_source": "diagnoses_all",
        "agg_funcs": [
            CountAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "ord_medicine_poly_pharmacy",
        "agg_funcs": [
            CountAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "ord_medicine_poly_pharmacy_since_diagnosis",
        "agg_funcs": [
            CountAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "diagnoses_all_comorbidity",
        "agg_funcs": [
            CountAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "blood_tests_all",
        "agg_funcs": [
            CountAggregator(),
            SumAggregator(),
            LatestAggregator(timestamp_col_name="timestamp"),
            # MaxAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "PERSIMUNE_leukocytes",
        "agg_funcs": [
            MeanAggregator(),
            SlopeAggregator(timestamp_col_name="timestamp"),
            MaxAggregator(),
            MinAggregator(),
            CountAggregator(),
            VarianceAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "PERSIMUNE_microbiology_analysis",
        "agg_funcs": [CountAggregator(), MaxAggregator()],
        "proportion": 0.1,
    },
    {
        "data_source": "PERSIMUNE_microbiology_culture",
        "agg_funcs": [CountAggregator(), MaxAggregator()],
        "proportion": 0.1,
    },
    {
        "data_source": "pathology_concat",
        "agg_funcs": [CountAggregator()],
        "proportion": 0.1,
    },
]
