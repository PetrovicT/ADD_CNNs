import pandas as pd
pd.set_option('display.max_rows', None)


def upload_excel_number_of_subject(file_path, sheet_name, class_name):
    xl_file = pd.ExcelFile(file_path)
    dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    df = dfs[sheet_name]
    column_names = list(df.columns)
    df = df.loc[:, column_names]
    # print(df.shape)  # print rows, cols

    unique_subjects = []
    index_research_group_column = 0
    index_subject_id = 0
    for index in range(len(column_names)):
        if column_names[index] == "Research Group" or column_names[index] == "Group":
            index_research_group_column = index
        if column_names[index] == "Subject ID" or column_names[index] == "Subject":
            index_subject_id = index

    for row in df.to_numpy():
        if row[index_subject_id] not in unique_subjects and row[index_research_group_column] == class_name:
            unique_subjects.append(row[index_subject_id])

    print("File: " + file_path)
    print("Class: " + class_name + ", number of subjects: " + str(len(unique_subjects)))


def split_string_find_image_id(image_name_string):
    substrings = image_name.split("_")
    image_id_length = len(substrings)
    full_image_id = substrings[image_id_length - 1]  # Iid
    # full_image_id[1::]  # only id, without letter I
    return full_image_id


def get_image_group(image_name_string, file_path, sheet_name):
    image_id = split_string_find_image_id(image_name_string)
    xl_file = pd.ExcelFile(file_path)
    dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    df = dfs[sheet_name]
    column_names = list(df.columns)
    df = df.loc[:, column_names]
    # print(df.shape)  # print rows, cols

    index_research_group_column = 0
    index_image_id = 0
    for index in range(len(column_names)):
        if column_names[index] == "Research Group" or column_names[index] == "Group":
            index_research_group_column = index
        if column_names[index] == "Image Data ID":
            index_image_id = index

    for row in df.to_numpy():
        if row[index_image_id] == image_id:
            return row[index_research_group_column]


if __name__ == '__main__':
    path_file = './tables/preprocessed_OR.xlsx'
    # upload_excel_number_of_subject(path_file, 'preprocessed_OR', "AD")
    # upload_excel_number_of_subject(path_file, 'preprocessed_OR', "CN")

    image_name = "ADNI_002_S_0295_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070319114336780_S13407_I45112"
    image_group = get_image_group(image_name, path_file, 'preprocessed_OR')
    print(image_group)
