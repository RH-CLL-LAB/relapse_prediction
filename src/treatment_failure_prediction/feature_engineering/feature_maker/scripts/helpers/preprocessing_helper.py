from helpers.constants import *
import pandas as pd
import datetime


def change_month(string: str) -> str:
    """Change format of months from strings to numbers

    Args:
        string (str): String with month (e.g. "JAN")

    Returns:
        str: Formatted string. If not possible, returns original string.
    """
    new_value = MONTH_MAPPING.get(string)
    if new_value:
        return new_value
    else:
        return string


def ATC_AB(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Converts to categorical from codes

    Args:
        data (pd.DataFrame): Dataframe containing columns
        column (str): Column containing values

    Returns:
        pd.DataFrame: Dataframe with added columns "AB_GROUP", "AB_GROUP_12" and "AB_DETAILED"
    """
    data["AB_GROUP"] = pd.NA
    for list_of_values, string in zip(
        [NARROW, BROAD, ANTIVIRAL, ANTIHELMINITICS, ANTIMYCOTICS],
        [
            "Narrow antibiotics",
            "Broad antibiotics",
            "Antivirals",
            "Antihelminitics",
            "Antimycotics",
        ],
    ):
        data.loc[
            (data["AB_GROUP"].isna())
            & (
                data[column].str.contains(
                    "|".join(list_of_values), regex=True, na=False
                )
            ),
            "AB_GROUP",
        ] = string
    for list_of_values, string in zip(
        [
            ["^J01A"],
            ["^J01C"],
            ["^J01E"],
            ["^J01DB", "^J01DC", "^J01DD"],
            ["^J01DH"],
            ["^J01F"],
            ["^J01M"],
            ["^J02A"],
            ["^J05A"],
            ["^P01A"],
            ["^P02C"],
            ["^J01X"],
        ],
        [
            "Tetracyclines",
            "Penicillins",
            "Sulfonamides",
            "Cephalosporins",
            "Carbapenems",
            "Macrolides",
            "Quinolone",
            "Antimycotics",
            "Antivirals",
            "Antiprotozoals",
            "Antihelminthics",
            "Other antibacterials",
        ],
    ):
        data.loc[
            (data["AB_GROUP_12"].isna())
            & (
                data[column].str.contains(
                    "|".join(list_of_values), regex=True, na=False
                )
            ),
            "AB_GROUP_12",
        ] = string
    data["AB_GROUP"] = pd.Categorical(
        data["AB_GROUP"],
        categories=[
            "Narrow antibiotics",
            "Broad antibiotics",
            "Antivirals",
            "Antimycotics",
            "Antihelminitics",
        ],
    )
    data["AB_GROUP_12"] = pd.Categorical(
        data["AB_GROUP_12"],
        [
            "Tetracyclines",
            "Penicillins",
            "Sulfonamides",
            "Cephalosporins",
            "Carbapenems",
            "Macrolides",
            "Quinolone",
            "Other antibacterials",
            "Antimycotics",
            "Antivirals",
            "Antiprotozoals",
            "Antihelminthics",
        ],
        ordered=True,
    )

    data["AB_DETAILED"] = data[column].apply(
        lambda x: PRECISE_ANTIBIOTICS_MAPPING.get(x)
    )

    return data


def ATC_opioids(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Converts to categorical from codes

    Args:
        data (pd.DataFrame): Dataframe containing column
        column (str): Column containing values

    Returns:
        pd.DataFrame: Dataframe with added column "ATC_OPIODS"
    """
    data["ATC_opioids"] = data[column].apply(lambda x: OPIODS_MAPPING.get(x))
    return data


def ATC_hypertensives(data: pd.DataFrame, column: str):
    """Converts from HTN_GROUP codes to categorical values

    Args:
        data (pd.DataFrame): Dataframe containing column
        column (str): Column containing values

    Returns:
        pd.DataFrame: Dataframe with added column "HTN_GROUP"
    """
    data["HTN_GROUP"] = pd.NA
    for list_of_values, string in zip(
        [
            ["^C03"],
            ["^C07A"],
            ["^C08"],
            ["^C09"],
            ["^C09XA"],
            ["^C02AB"],
            ["^C02AC"],
            ["^C02CA", "^G04CA"],
            ["^C02DB"],
            ["^C02DD"],
        ],
        [
            "Diuretics",
            "Beta blockers",
            "Calcium chanel blockers",
            "ACEI/ARB",
            "Renim inhibitors",
            "Methyldopa",
            "Moxonidin",
            "Alpha blockers",
            "Hydralazin",
            "Nitroprussid",
        ],
    ):
        data.loc[
            (data["HTN_GROUP"].isna())
            & (
                data[column].str.contains(
                    "|".join(list_of_values), regex=True, na=False
                )
            ),
            "HTN_GROUP",
        ] = string
    return data


def date_from_origin_unix(number_of_days: str) -> datetime.datetime:
    """Converts string indicating date to datetime.date

    Args:
        number_of_days (str): string containing information about str

    Returns:
        datetime.date: datetime object
    """
    return datetime.date(year=1970, month=1, day=1) + datetime.timedelta(
        days=number_of_days
    )


def date_from_timestamp(timestamp: int) -> datetime.date:
    """Converts integer indicating date to datetime.date

    Args:
        timestamp (int): integer containing the timestamp of the date

    Returns:
        datetime.date: Datetime date object
    """
    return datetime.date.fromtimestamp(timestamp)


def date_from_date(date: datetime.date) -> datetime.date:
    """Converts datetime date to date

    Args:
        date (datetime.date): Date from column

    Returns:
        datetime.date: Date
    """
    return_date = pd.to_datetime(date, errors="coerce")
    if return_date:
        return return_date.date()
    else:
        return pd.NA


DATE_CONVERTER = {
    "RKKP_CLL": date_from_date,
    "RKKP_LYFO": date_from_date,
    "RKKP_DaMyDa": date_from_date,
    "PERSIMUNE_biochemistry": date_from_date,
    "PERSIMUNE_microbiology_analysis": date_from_date,
    "PERSIMUNE_microbiology_culture": date_from_date,
    "PERSIMUNE_microbiology_extra": date_from_date,
    "PERSIMUNE_microbiology_microscopy": date_from_date,
    "SDS_ekokur": date_from_origin_unix,
    "SDS_epikur": date_from_origin_unix,
    "SDS_forloeb": date_from_origin_unix,
    "SDS_indberetningmedpris": date_from_timestamp,
    "SDS_kontakter": date_from_origin_unix,
    "SDS_lab_forsker": date_from_origin_unix,
    "SDS_pato": date_from_origin_unix,
    "SDS_t_konk_ny": date_from_origin_unix,
    "SDS_t_mikro_ny": date_from_origin_unix,
    "SDS_procedurer_kirurgi": date_from_origin_unix,
    "SDS_procedurer_andre": date_from_origin_unix,
    "SDS_forloebsmarkoerer": date_from_origin_unix,
    "SDS_resultater": date_from_origin_unix,
    "SDS_t_adm": date_from_origin_unix,
    "SDS_t_udtilsgh": date_from_origin_unix,
    "SDS_t_dodsaarsag_2": date_from_origin_unix,
    "SDS_t_tumor": date_from_origin_unix,
    "SDS_diagnoser": date_from_origin_unix,
    "SP_Aktive_Problemliste_Diagnoser": date_from_date,
    "SP_AlleProvesvar": date_from_date,
    "SP_OrdineretMedicin": date_from_date,
    "SP_ADT_haendelser": date_from_timestamp,
    "SP_Administreret_Medicin": date_from_timestamp,
    "SP_Behandlingsplaner_del1": date_from_timestamp,
    "SP_Behandlingsplaner_del2": date_from_timestamp,
    "SP_Bloddyrkning_del1": date_from_timestamp,
    "SP_Bloddyrkning_del2": date_from_timestamp,
    "SP_Bloddyrkning_del3": date_from_timestamp,
    "SP_ItaOphold": date_from_timestamp,
    "SP_Journalnotater_del1": date_from_timestamp,
    "SP_Journalnotater_del2": date_from_timestamp,
    "SP_Social_Hx": date_from_timestamp,
    "SP_VitaleVaerdier": date_from_timestamp,
    "SP_Flytningshistorik": date_from_timestamp,
    "SP_Behandlingskontakter_diagnoser": date_from_timestamp,
    "view_sds_t_adm_t_diag": date_from_date,
    "view_sds_t_adm_t_sksopr": date_from_date,
    "view_sds_t_adm_t_sksube": date_from_date,
    "LAB_Flowcytometry": date_from_date,
    "LAB_IGHVIMGT": date_from_date,
}
