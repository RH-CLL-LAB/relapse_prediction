"""
sql_helper.py — helpers for connecting to the Postgres DB and loading tables.

NOTE:
- Behaviour is kept identical to the original version.
- Global variables IMPORT_PUBLIC, LAB_PUBLIC, CORE_PUBLIC, etc. are still
  initialised at import time by querying the database, and OPTIONS_DICTIONARY
  is still mutated in-place in the same order.
"""

import psycopg2
import psycopg2.extras
import pandas as pd

from lyfo_treatment_failure_prediction.helpers.constants import OPTIONS_DICTIONARY


def connect_to_db(options_dictionary: dict) -> psycopg2.connect:
    """Given a dictionary containing the configurations for the connection,
    connect to the database.

    Args:
        options_dictionary (dict): dictionary containing the configurations.

    Returns:
        psycopg2.connection: connection to the database.
    """
    conn = psycopg2.connect(
        database=options_dictionary["database"],
        host=options_dictionary["host"],
        user=options_dictionary["user"],
        password=options_dictionary["password"],
        port=options_dictionary["port"],
        options=options_dictionary["options"],
    )
    return conn


def query_connection(
    conn: psycopg2.connect,
    query: str,
    print_output: bool = False,
    get_columns: bool = False,
):
    """Given a connection, query the database and return the output.

    Args:
        conn (psycopg2.connection): Connection to the database.
        query (str): String containing the query to be executed.
        print_output (bool, optional): Whether to print the output. Defaults to False.
        get_columns (bool, optional): Whether to also return column names. Defaults to False.

    Returns:
        list or (list, list): query result, optionally with column names.
    """
    cursor = conn.cursor()

    cursor.execute(query)

    output = cursor.fetchall()

    if print_output:
        print(output)

    if get_columns:
        col_names = [desc[0] for desc in cursor.description]
        conn.close()
        return col_names, output

    conn.close()

    return output


def get_tables(connection: psycopg2.connect) -> list:
    """Return list of dicts with schemas and table names in the DB.

    Args:
        connection (psycopg2.connection): Connection to the database.

    Returns:
        list: list of dictionaries (table_schema, table_name).
    """
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cursor.execute(
        """SELECT table_schema, table_name
           FROM information_schema.tables
           WHERE table_schema != 'pg_catalog'
             AND table_schema != 'information_schema'
             AND table_type='BASE TABLE'
           ORDER BY table_schema, table_name"""
    )

    tables = cursor.fetchall()

    cursor.close()

    return tables


def connect_and_query(
    options_dictionary: dict,
    query: str,
    print_output: bool = False,
    get_columns: bool = False,
):
    """Helper to connect and then run a single query."""
    conn = connect_to_db(options_dictionary=options_dictionary)
    output = query_connection(
        conn=conn,
        query=query,
        print_output=print_output,
        get_columns=get_columns,
    )
    return output


# ---------------------------------------------------------------------------
# Global initialization — identical behaviour to original
# ---------------------------------------------------------------------------

# Start with OPTIONS_DICTIONARY as imported from constants
# and progressively mutate it as in the original script.

IMPORT_PUBLIC = sorted(
    [
        table_name[0]
        for table_name in connect_and_query(
            options_dictionary=OPTIONS_DICTIONARY,
            query=(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ),
            print_output=False,
            get_columns=False,
        )
    ]
)
dictionary_of_dictionaries = {"IMPORT_PUBLIC": OPTIONS_DICTIONARY.copy()}

OPTIONS_DICTIONARY["options"] = "-c search_path=laboratory"

LAB_PUBLIC = sorted(
    [
        table_name[0]
        for table_name in connect_and_query(
            options_dictionary=OPTIONS_DICTIONARY,
            query=(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'laboratory'"
            ),
            print_output=False,
            get_columns=False,
        )
    ]
)
dictionary_of_dictionaries["LAB_PUBLIC"] = OPTIONS_DICTIONARY.copy()

OPTIONS_DICTIONARY["database"] = "core"
OPTIONS_DICTIONARY["options"] = "-c search_path=public"

CORE_PUBLIC = sorted(
    [
        table_name[0]
        for table_name in connect_and_query(
            options_dictionary=OPTIONS_DICTIONARY,
            query=(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ),
            print_output=False,
            get_columns=False,
        )
    ]
)
dictionary_of_dictionaries["CORE_PUBLIC"] = OPTIONS_DICTIONARY.copy()

OPTIONS_DICTIONARY["database"] = "import"
OPTIONS_DICTIONARY["options"] = "-c search_path=_tables"

IMPORT_TABLES = sorted(
    [
        table_name[0]
        for table_name in connect_and_query(
            options_dictionary=OPTIONS_DICTIONARY,
            query=(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = '_tables'"
            ),
            print_output=False,
            get_columns=False,
        )
    ]
)
dictionary_of_dictionaries["IMPORT_TABLES"] = OPTIONS_DICTIONARY.copy()

OPTIONS_DICTIONARY["options"] = "-c search_path=_lookup_tables"

IMPORT_LOOKUP_TABLES = sorted(
    [
        table_name[0]
        for table_name in connect_and_query(
            options_dictionary=OPTIONS_DICTIONARY,
            query=(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = '_lookup_tables'"
            ),
            print_output=False,
            get_columns=False,
        )
    ]
)
dictionary_of_dictionaries["IMPORT_LOOKUP_TABLES"] = OPTIONS_DICTIONARY.copy()

OPTIONS_DICTIONARY["database"] = "core"
OPTIONS_DICTIONARY["options"] = "-c search_path=_lookup_tables"

CORE_LOOKUP_TABLES = sorted(
    [
        table_name[0]
        for table_name in connect_and_query(
            options_dictionary=OPTIONS_DICTIONARY,
            query=(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = '_lookup_tables'"
            ),
            print_output=False,
            get_columns=False,
        )
    ]
)
dictionary_of_dictionaries["CORE_LOOKUP_TABLES"] = OPTIONS_DICTIONARY.copy()


# ---------------------------------------------------------------------------
# Cohort helpers
# ---------------------------------------------------------------------------

def get_cohort_string_from_data(data: pd.DataFrame, id_col: str = "patientid") -> str:
    """Helper function for creating string to limit loading to only include cohort.

    Args:
        data (pd.DataFrame): dataframe containing the ids of interest.
        id_col (str, optional): Name of id-column. Defaults to "patientid".

    Returns:
        str: string of ids for querying, e.g. "(1, 2, 3)".
    """
    return f"({', '.join(data[id_col].astype(str).values)})"


def get_cohort_string_from_data_as_strings(
    data: pd.DataFrame, id_col: str = "patientid"
) -> str:
    """Helper function for creating string to limit loading to only include cohort
    (with ids as strings).

    NOTE: This keeps the original, slightly odd formatting exactly as-is.

    Args:
        data (pd.DataFrame): dataframe containing the ids of interest.
        id_col (str, optional): Name of id-column. Defaults to "patientid".

    Returns:
        str: string of ids for querying.
    """
    return f'("{"", "".join(data[id_col].astype(str).values)}"")'


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_data_from_table(
    table_name: str,
    subset_columns: list = [],
    limit: int = 0,
    cohort_column: str = "patientid",
    cohort: str = "",
) -> pd.DataFrame:
    """Given a table name, load a subset of columns and an optional limit.

    Args:
        table_name (str): Name of datasource.
        subset_columns (list, optional): Subset of columns. Defaults to [], meaning all.
        limit (int, optional): Row limit. Defaults to 0 (all rows).
        cohort_column (str, optional): Column used for cohort filtering. Defaults to "patientid".
        cohort (str, optional): String used in WHERE ... IN ( ... ). Defaults to "".

    Returns:
        pd.DataFrame: Loaded dataframe, or None if table not found.
    """
    if limit:
        limit_string = f" LIMIT {limit}"
    else:
        limit_string = ""

    relevant_table = [
        sql_table_str
        for sql_table, sql_table_str in zip(
            [
                IMPORT_PUBLIC,
                CORE_PUBLIC,
                IMPORT_TABLES,
                IMPORT_LOOKUP_TABLES,
                CORE_LOOKUP_TABLES,
                LAB_PUBLIC,
            ],
            [
                "IMPORT_PUBLIC",
                "CORE_PUBLIC",
                "IMPORT_TABLES",
                "IMPORT_LOOKUP_TABLES",
                "CORE_LOOKUP_TABLES",
                "LAB_PUBLIC",
            ],
        )
        if table_name in sql_table
    ]

    if relevant_table:
        if subset_columns:
            joined_string_of_columns = ", ".join([f'"{x}"' for x in subset_columns])
            query = f'SELECT {joined_string_of_columns} FROM "{table_name}"'
        else:
            query = f'SELECT * FROM "{table_name}"'

        if cohort:
            cohort = f' WHERE "{cohort_column}" IN {cohort}'
        query = query + cohort + limit_string + ";"

        col_names, output = connect_and_query(
            options_dictionary=dictionary_of_dictionaries[relevant_table[0]],
            query=query,
            print_output=False,
            get_columns=True,
        )
        dataframe = pd.DataFrame(output, columns=col_names)
        return dataframe
    else:
        print("Dataframe not found!")
        return None


def join_with_pathology_table(table_name: str) -> pd.DataFrame:
    """Helper function for joining the pathology tables.

    Args:
        table_name (str): Name of data source.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    query = f"""SELECT id, date
    FROM "{table_name}" konk
    INNER JOIN (
        SELECT MIN(pato.patientid) id,
               MIN(pato.d_rekvdato) date,
               pato.k_inst,
               pato.k_rekvnr
        FROM "SDS_pato" AS pato
        GROUP BY pato.k_inst, pato.k_rekvnr
    ) AS joined_table
    ON konk.k_rekvnr = joined_table.k_rekvnr
    AND konk.k_inst = joined_table.k_inst
    """

    data = connect_and_query(
        options_dictionary=dictionary_of_dictionaries["IMPORT_PUBLIC"],
        query=query,
        get_columns=True,
    )
    dataframe = pd.DataFrame(data=data[1], columns=data[0])
    dataframe["data_source"] = table_name
    return dataframe


def download_and_rename_data(
    table_name: str, config_dict: dict, cohort: str = "", cohort_column: str = "patientid"
) -> pd.DataFrame:
    """Downloads and renames the data queried from SQL.

    Args:
        table_name (str): Name of the table from SQL.
        config_dict (dict): configuration dict for downloading.
        cohort (str, optional): Cohort string for WHERE ... IN. Defaults to "".
        cohort_column (str, optional): Column used for cohort filtering. Defaults to "patientid".

    Returns:
        pd.DataFrame: dataframe containing the relevant information.
    """
    columns = list(config_dict[table_name].keys())
    data = load_data_from_table(
        table_name=table_name,
        subset_columns=columns,
        cohort=cohort,
        cohort_column=cohort_column,
    )

    data = data.rename(columns=config_dict[table_name])
    data["data_source"] = table_name
    return data


def merge_data_frames_without_duplicates(
    left_dataframe: pd.DataFrame, right_dataframe: pd.DataFrame, on: str
) -> pd.DataFrame:
    """Merge two dataframes while avoiding overlapping columns.

    Args:
        left_dataframe (pd.DataFrame): Left dataframe.
        right_dataframe (pd.DataFrame): Right dataframe.
        on (str): column to merge on.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    non_overlapping_columns = [
        x for x in right_dataframe.columns if x not in left_dataframe.columns
    ]
    non_overlapping_columns.append(on)
    subset_of_right = right_dataframe[non_overlapping_columns]
    return left_dataframe.merge(subset_of_right, on=on).reset_index(drop=True)


__all__ = [
    "connect_to_db",
    "query_connection",
    "get_tables",
    "connect_and_query",
    "IMPORT_PUBLIC",
    "LAB_PUBLIC",
    "CORE_PUBLIC",
    "IMPORT_TABLES",
    "IMPORT_LOOKUP_TABLES",
    "CORE_LOOKUP_TABLES",
    "dictionary_of_dictionaries",
    "get_cohort_string_from_data",
    "get_cohort_string_from_data_as_strings",
    "load_data_from_table",
    "join_with_pathology_table",
    "download_and_rename_data",
    "merge_data_frames_without_duplicates",
]
