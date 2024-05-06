from helpers.sql_helper import *
from helpers.preprocessing_helper import *
from helpers.constants import *
from tqdm import tqdm

print("Downloading LPR3 tables")
list_of_lpr_three_data = [
    download_and_rename_data(table_name=table_name, config_dict=LPR_THREE_TABLES)
    for table_name in tqdm(LPR_THREE_TABLES)
]

adm_merge_df = list_of_lpr_three_data[-1][
    [x for x in list_of_lpr_three_data[-1].columns if x != "data_source"]
]

contact_element = list_of_lpr_three_data.pop()

list_of_lpr_three_merged_data = []

for lpr_three_data in tqdm(list_of_lpr_three_data):
    contact_df, course_df = lpr_three_data.copy(), lpr_three_data.copy()
    if "course_id" in list(lpr_three_data.columns) and "contact_id" in list(
        lpr_three_data.columns
    ):
        # logic being that there will be no overlap between these, and we get all valid data points
        contact_df = lpr_three_data[lpr_three_data["contact_id"].notna()].reset_index(
            drop=True
        )
        course_df = lpr_three_data[
            (lpr_three_data["course_id"].notna())
            & (lpr_three_data["contact_id"].isna())
        ].reset_index(drop=True)

    if "course_id" in list(lpr_three_data.columns):
        list_of_lpr_three_merged_data.append(
            merge_data_frames_without_duplicates(
                course_df, adm_merge_df, on="course_id"
            )
        )

    if "contact_id" in list(lpr_three_data.columns):
        list_of_lpr_three_merged_data.append(
            merge_data_frames_without_duplicates(
                contact_df, adm_merge_df, on="contact_id"
            )
        )

list_of_lpr_three_merged_data.append(contact_element)
