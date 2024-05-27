from helpers.sql_helper import *
from helpers.preprocessing_helper import *
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from data_processing.wide_data import *

# for embeddings, we probably need to filter the data to be before

journal_dict = {
    "SP_Journalnotater_Del1": {
        "patientid": "patientid",
        "ydelsesdato_dato_tid": "timestamp",
        "forfatter_type": "variable_code",
        "notat_text": "value",
    }
}

journal_notes = download_and_rename_data(
    "SP_Journalnotater_Del1", journal_dict, cohort=lyfo_cohort
)

prediction_times = WIDE_DATA[["patientid", "date_treatment_1st_line"]]

merged_notes = journal_notes.merge(prediction_times)

merged_notes["timestamp"] = pd.to_datetime(
    merged_notes["timestamp"], errors="coerce", unit="s"
)

merged_notes = merged_notes[
    merged_notes["timestamp"] < merged_notes["date_treatment_1st_line"]
].reset_index(drop=True)

merged_notes["patientid"] = merged_notes["patientid"].astype(int)

stop_words = [
    "01",
    "03",
    "04",
    "50",
    "05",
    "06",
    "17",
    "25",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "18",
    "19",
    "20",
    "30",
    "at",
    "da",
    "de",
    "den",
    "det",
    "dette",
    "en",
    "et",
    "nej",
    "sin",
    "som",
    "vi",
    "vil",
]


# define function to embed text and return a dataframe
def embed_text_to_df(text: list[str]) -> pl.DataFrame:
    tfidf_model = TfidfVectorizer(
        max_features=200, stop_words=stop_words, max_df=0.5, ngram_range=(1, 3)
    )
    embeddings = tfidf_model.fit_transform(text)
    return pl.DataFrame(
        embeddings.toarray(), schema=tfidf_model.get_feature_names_out().tolist()
    )


# embed text
embedded_text = embed_text_to_df(text=merged_notes["value"].to_list())
# drop the text column from the original dataframe
metadata_only = pl.DataFrame(merged_notes[["patientid", "timestamp"]])

# concatenate the metadata and the embedded text
embedded_text_with_metadata = pl.concat(
    [metadata_only, embedded_text], how="horizontal"
)
