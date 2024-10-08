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
        "data_source": "medicine_general",
        "agg_funcs": [
            CountAggregator(),
            # SumAggregator(),
            # LatestAggregator(timestamp_col_name="timestamp"),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "medicine_1825_days_count",
        "agg_funcs": [
            CountAggregator(),
            SumAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "medicine_365_days_count",
        "agg_funcs": [
            CountAggregator(),
            SumAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "medicine_90_days_count",
        "agg_funcs": [
            CountAggregator(),
            SumAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "medicine_1825_days_cumulative",
        "agg_funcs": [
            SumAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "medicine_365_days_cumulative",
        "agg_funcs": [
            SumAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "medicine_90_days_cumulative",
        "agg_funcs": [
            SumAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "LAB_IGHVIMGT",
        "agg_funcs": [
            # CountAggregator(),
            # SumAggregator(),
            LatestAggregator(timestamp_col_name="timestamp"),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "LAB_BIOBANK_SAMPLES",
        "agg_funcs": [
            # CountAggregator(),
            # SumAggregator(),
            LatestAggregator(timestamp_col_name="timestamp"),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "LAB_Flowcytometry",
        "agg_funcs": [
            # CountAggregator(),
            # SumAggregator(),
            LatestAggregator(timestamp_col_name="timestamp"),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "labmeasurements",
        "agg_funcs": [
            CountAggregator(),
            # SumAggregator(),
            MaxAggregator(),
            SlopeAggregator(timestamp_col_name="timestamp"),
            MinAggregator(),
            LatestAggregator(timestamp_col_name="timestamp"),
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
    {"data_source": "SDS_pato", "agg_funcs": [CountAggregator()], "proportion": 0.1},
    {
        "data_source": "diagnoses_all",
        "agg_funcs": [
            CountAggregator(),
            LatestAggregator(timestamp_col_name="timestamp"),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "poly_pharmacy",
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
        "data_source": "SP_SocialHx",
        "agg_funcs": [LatestAggregator(timestamp_col_name="timestamp")],
        "proportion": 0.1,
    },
    {
        "data_source": "SP_Bloddyrkning_Del1",
        "agg_funcs": [
            CountAggregator(),
            SumAggregator(),
            LatestAggregator(timestamp_col_name="timestamp"),
            MaxAggregator(),
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
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "PERSIMUNE_radiology",
        "agg_funcs": [
            CountAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "SP_BilleddiagnostiskeUndersøgelser_Del1",
        "agg_funcs": [
            CountAggregator(),
            LatestAggregator(timestamp_col_name="timestamp"),
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
        "data_source": "SP_VitaleVaerdier",
        "agg_funcs": [
            CountAggregator(),
            MeanAggregator(),
            MaxAggregator(),
            LatestAggregator(timestamp_col_name="timestamp"),
            SlopeAggregator(timestamp_col_name="timestamp"),
            MinAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "gene_alterations",
        "agg_funcs": [CountAggregator(), SumAggregator()],
        "proportion": 0.01,
    },
]

# feature_specs = [
#     {
#         "data_source": "medicine_general",
#         "agg_funcs": [
#             CountAggregator(),
#             # SumAggregator(),
#             # LatestAggregator(timestamp_col_name="timestamp"),
#         ],
#         "proportion": 0.1,
#     },
#     {
#         "data_source": "medicine_1080_days_count",
#         "agg_funcs": [
#             CountAggregator(),
#             SumAggregator(),
#         ],
#         "proportion": 0.1,
#     },
#     {
#         "data_source": "medicine_365_days_count",
#         "agg_funcs": [
#             CountAggregator(),
#             SumAggregator(),
#         ],
#         "proportion": 0.1,
#     },
#     {
#         "data_source": "medicine_90_days_count",
#         "agg_funcs": [
#             CountAggregator(),
#             SumAggregator(),
#         ],
#         "proportion": 0.1,
#     },
#     {
#         "data_source": "medicine_1080_days_cumulative",
#         "agg_funcs": [
#             SumAggregator(),
#         ],
#         "proportion": 0.1,
#     },
#     {
#         "data_source": "medicine_365_days_cumulative",
#         "agg_funcs": [
#             SumAggregator(),
#         ],
#         "proportion": 0.1,
#     },
#     {
#         "data_source": "medicine_90_days_cumulative",
#         "agg_funcs": [
#             SumAggregator(),
#         ],
#         "proportion": 0.1,
#     },
#     {
#         "data_source": "LAB_IGHVIMGT",
#         "agg_funcs": [
#             # CountAggregator(),
#             # SumAggregator(),
#             LatestAggregator(timestamp_col_name="timestamp"),
#         ],
#         "proportion": 0.001,
#     },
#     {
#         "data_source": "LAB_BIOBANK_SAMPLES",
#         "agg_funcs": [
#             # CountAggregator(),
#             # SumAggregator(),
#             LatestAggregator(timestamp_col_name="timestamp"),
#         ],
#         "proportion": 0.001,
#     },
#     {
#         "data_source": "LAB_Flowcytometry",
#         "agg_funcs": [
#             # CountAggregator(),
#             # SumAggregator(),
#             LatestAggregator(timestamp_col_name="timestamp"),
#         ],
#         "proportion": 0.001,
#     },
#     {
#         "data_source": "labmeasurements",
#         "agg_funcs": [
#             CountAggregator(),
#             # SumAggregator(),
#             MaxAggregator(),
#             SlopeAggregator(timestamp_col_name="timestamp"),
#             MinAggregator(),
#             LatestAggregator(timestamp_col_name="timestamp"),
#         ],
#         "proportion": 0.001,
#     },
#     {
#         "data_source": "sks_referals",
#         "agg_funcs": [CountAggregator(), SumAggregator()],
#         "proportion": 0.1,
#     },
#     {
#         "data_source": "sks_at_the_hospital",
#         "agg_funcs": [CountAggregator(), SumAggregator()],
#         "proportion": 0.1,
#     },
#     {"data_source": "SDS_pato", "agg_funcs": [CountAggregator()], "proportion": 0.1},
#     {
#         "data_source": "diagnoses_all",
#         "agg_funcs": [
#             CountAggregator(),
#             LatestAggregator(timestamp_col_name="timestamp"),
#         ],
#         "proportion": 0.05,
#     },
#     {
#         "data_source": "diagnoses_all_comorbidity",
#         "agg_funcs": [
#             CountAggregator(),
#         ],
#         "proportion": 0.1,
#     },
#     {
#         "data_source": "SP_SocialHx",
#         "agg_funcs": [LatestAggregator(timestamp_col_name="timestamp")],
#         "proportion": 0.01,
#     },
#     {
#         "data_source": "SP_Bloddyrkning_Del1",
#         "agg_funcs": [
#             CountAggregator(),
#             SumAggregator(),
#             LatestAggregator(timestamp_col_name="timestamp"),
#             MaxAggregator(),
#         ],
#         "proportion": 0.01,
#     },
#     {
#         "data_source": "PERSIMUNE_leukocytes",
#         "agg_funcs": [
#             MeanAggregator(),
#             SlopeAggregator(timestamp_col_name="timestamp"),
#             MaxAggregator(),
#             MinAggregator(),
#             CountAggregator(),
#         ],
#         "proportion": 0.05,
#     },
#     {
#         "data_source": "SP_BilleddiagnostiskeUndersøgelser_Del1",
#         "agg_funcs": [
#             CountAggregator(),
#             LatestAggregator(timestamp_col_name="timestamp"),
#         ],
#         "proportion": 0.01,
#     },
#     {
#         "data_source": "PERSIMUNE_microbiology_analysis",
#         "agg_funcs": [CountAggregator(), MaxAggregator()],
#         "proportion": 0.01,
#     },
#     {
#         "data_source": "PERSIMUNE_microbiology_culture",
#         "agg_funcs": [CountAggregator(), MaxAggregator()],
#         "proportion": 0.01,
#     },
#     {
#         "data_source": "SP_VitaleVaerdier",
#         "agg_funcs": [
#             CountAggregator(),
#             MeanAggregator(),
#             MaxAggregator(),
#             LatestAggregator(timestamp_col_name="timestamp"),
#             SlopeAggregator(timestamp_col_name="timestamp"),
#             MinAggregator(),
#         ],
#         "proportion": 0.1,
#     },
# ]
