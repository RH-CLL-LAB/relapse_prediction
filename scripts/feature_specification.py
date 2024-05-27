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

# NOTE: Ask Casper to include data source in views
# NOTE: Include year of treatment as a feature in wide_data
# NOTE: Use epikur and ekokur instead of RECEPTDATA_clean
# NOTE: Is EPIKUR and EKOKUR the same....?????????
# NOTE: SP_lab_forsker should be in included

feature_specs = [
    # {
    #     "data_source": "medicine",
    #     "agg_funcs": [CountAggregator(), SumAggregator()],
    #     "proportion": 0.1,
    # },
    {
        "data_source": "adm_medicine",
        "agg_funcs": [
            CountAggregator(),
            SumAggregator(),
            SlopeAggregator(timestamp_col_name="timestamp"),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "SP_OrdineretMedicin",
        "agg_funcs": [CountAggregator(), SumAggregator()],
        "proportion": 0.1,
    },
    {
        "data_source": "SDS_epikur",
        "agg_funcs": [CountAggregator(), SumAggregator()],
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
        "data_source": "laboratorymeasurements",
        "agg_funcs": [
            LatestAggregator(timestamp_col_name="timestamp"),
            CountAggregator(),
            MeanAggregator(),
            MaxAggregator(),
            MinAggregator(),
            SlopeAggregator(timestamp_col_name="timestamp"),
            SumAggregator(),
            VarianceAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "SP_AlleProvesvar",
        "agg_funcs": [
            LatestAggregator(timestamp_col_name="timestamp"),
            CountAggregator(),
            MeanAggregator(),
            MaxAggregator(),
            MinAggregator(),
            SlopeAggregator(timestamp_col_name="timestamp"),
            SumAggregator(),
            VarianceAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "laboratorymeasurements_concat",  # does this actually make sense?
        "agg_funcs": [
            LatestAggregator(timestamp_col_name="timestamp"),
            CountAggregator(),
            MeanAggregator(),
            MaxAggregator(),
            MinAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "RECEPTDATA_CLEAN",
        "agg_funcs": [
            CountAggregator(),
            SumAggregator(),
            SlopeAggregator(timestamp_col_name="timestamp"),
        ],
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
        "data_source": "SP_SocialHx",
        "agg_funcs": [LatestAggregator(timestamp_col_name="timestamp")],
        "proportion": 0.1,
    },
    {
        "data_source": "SP_Bloddyrkning_Del1",
        "agg_funcs": [CountAggregator()],
        "proportion": 0.1,
    },
    # {
    #    "data_source": "LYFO_AKI",
    #    "agg_funcs": [MaxAggregator()],
    #    "proportion": 0.0,
    # },
    {
        "data_source": "PERSIMUNE_biochemistry",
        "agg_funcs": [
            CountAggregator(),
            MeanAggregator(),
            SlopeAggregator(timestamp_col_name="timestamp"),
            LatestAggregator(timestamp_col_name="timestamp"),
            MaxAggregator(),
            MinAggregator(),
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
        "data_source": "diagnosis_all_comorbidity",
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
        "data_source": "view_laboratorymeasuments_c_groups",
        "agg_funcs": [
            CountAggregator(),
        ],
        "proportion": 0.1,
    },
    # {
    #     "data_source": "medicine_days",
    #     "agg_funcs": [
    #         CountAggregator(),
    #         MeanAggregator(),
    #         MaxAggregator(),
    #         SumAggregator(),
    #     ],
    #     "proportion": 0.1,
    # },
    {
        "data_source": "adm_medicine_days",
        "agg_funcs": [
            CountAggregator(),
            MeanAggregator(),
            MaxAggregator(),
            SumAggregator(),
        ],
        "proportion": 0.1,
    },
    {
        "data_source": "SP_OrdineretMedicin_days",
        "agg_funcs": [
            CountAggregator(),
            MeanAggregator(),
            MaxAggregator(),
            SumAggregator(),
        ],
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
]
