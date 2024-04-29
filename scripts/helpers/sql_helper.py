import psycopg2
import psycopg2.extras
import pandas as pd
from helpers.constants import OPTIONS_DICTIONARY


def connect_to_db(options_dictionary: dict) -> psycopg2.connect:
    """Given a dictionary containing the configurations for the connenction,
    connect to the database

    Args:
        options_dictionary (dict): dictionary containing the configurations

    Returns:
        psycopg2.connection: connection to the database
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
    """Given a connection, query the database and return the output

    Args:
        conn (psycopg2.connection): Connection to the database
        query (str): String containing the query to be executed
        print_output (bool, optional): Boolean dictating whether to print the output. Defaults to False.
        get_columns (bool, optional): Boolean indicating whether to also return the column names. Defaults to False.

    Returns:
        list: list containing the result of the query
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
    """Create and return a list of dictionaries with the
    schemas and names of tables in the database
    connected to by the connection argument.

    Args:
        connection (psycopg2.connection): Connection to the database

    Returns:
        list: list of dictionaries
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
    options_dictionary: dict, query: str, print_output: bool = False, get_columns=False
):
    conn = connect_to_db(options_dictionary=options_dictionary)

    output = query_connection(
        conn=conn, query=query, print_output=print_output, get_columns=get_columns
    )
    return output


# use queries to list all the tables

IMPORT_PUBLIC = sorted(
    [
        table_name[0]
        for table_name in connect_and_query(
            options_dictionary=OPTIONS_DICTIONARY,
            query="SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",  # "select relname from pg_class where relkind='r' and relname !~ '^(pg_|sql_)';",
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
            query="SELECT table_name FROM information_schema.tables WHERE table_schema = 'laboratory'",  # "select relname from pg_class where relkind='r' and relname !~ '^(pg_|sql_)';",
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
            query="SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
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
            query="SELECT table_name FROM information_schema.tables WHERE table_schema = '_tables'",
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
            query="SELECT table_name FROM information_schema.tables WHERE table_schema = '_lookup_tables'",
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
            query="SELECT table_name FROM information_schema.tables WHERE table_schema = '_lookup_tables'",
            print_output=False,
            get_columns=False,
        )
    ]
)
dictionary_of_dictionaries["CORE_LOOKUP_TABLES"] = OPTIONS_DICTIONARY.copy()


def get_cohort_string_from_data(data: pd.DataFrame, id_col="patientid") -> str:
    """Helper function for creating string to limit loading to only include cohort

    Args:
        data (pd.DataFrame): dataframe containing the ids of interest
        id_col (str, optional): Name of id-column in the dataframe. Defaults to "patientid".

    Returns:
        str: string of ids for querying
    """
    return f"({', '.join(data[id_col].astype(str).values)})"

def get_cohort_string_from_data_as_strings(data: pd.DataFrame, id_col="patientid") -> str:
    """Helper function for creating string to limit loading to only include cohort

    Args:
        data (pd.DataFrame): dataframe containing the ids of interest
        id_col (str, optional): Name of id-column in the dataframe. Defaults to "patientid".

    Returns:
        str: string of ids for querying
    """
    return f'("{"", "".join(data[id_col].astype(str).values)}"")'


def load_data_from_table(
    table_name: str,
    subset_columns: list = [],
    limit: int = 0,
    cohort_column: str = "patientid",
    cohort: str = "",
) -> pd.DataFrame:
    """Given a name for a dataframe, load a subset of columns and a limit of rows.

    Args:
        table_name (str): Name of datasource
        subset_columns (list, optional): Subset of columns. Defaults to [], which gives all columns.
        limit (int, optional): Limit of rows to load. Defaults to 0, which gives all rows.

    Returns:
        pd.DataFrame: Loaded dataframe
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
        # print(f"Now loading from: {table_name}:")
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
    """Helper function for joining the pathology tables

    Args:
        table_name (str): Name of data source

    Returns:
        pd.DataFrame: Merged dataframe
    """
    query = f"""SELECT id, date
    FROM "{table_name}" konk
    INNER JOIN (SELECT MIN(pato.patientid) id, MIN(pato.d_rekvdato) date, pato.k_inst, pato.k_rekvnr FROM "SDS_pato" AS pato GROUP BY pato.k_inst, pato.k_rekvnr) as joined_table
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
    table_name: str, config_dict: dict, cohort="", cohort_column: str = "patientid"
) -> pd.DataFrame:
    """Downloads and renames the data queried from SQL

    Args:
        table_name (str): String which is the name of the table from SQL
        config_dict (dict): dictionary giving the configuration for downloading

    Returns:
        pd.DataFrame: dataframe containing the relvant information
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
    # df["date"] = df["date"].apply(DATE_CONVERTER.get(table_name))
    return data


def merge_data_frames_without_duplicates(
    left_dataframe: pd.DataFrame, right_dataframe: pd.DataFrame, on: str
) -> pd.DataFrame:
    """Merged two data sources while avoiding overlapping columns

    Args:
        left_dataframe (pd.DataFrame): Left Dataframe
        right_dataframe (pd.DataFrame): Right Dataframe
        on (str): column to merge on

    Returns:
        pd.DataFrame: Merged dataframe
    """
    non_overlapping_columns = [
        x for x in right_dataframe.columns if x not in left_dataframe.columns
    ]
    non_overlapping_columns.append(on)
    subset_of_right = right_dataframe[non_overlapping_columns]
    return left_dataframe.merge(subset_of_right, on=on).reset_index(drop=True)
