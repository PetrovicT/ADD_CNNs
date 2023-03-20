import pandas as pd
pd.set_option('display.max_rows', None)


def get_excel_number_of_subjects(file_path: str, sheet_name: str, class_name: str) -> int:
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

    num_of_subjects = len(unique_subjects)
    print("File: " + file_path)
    print("Class: " + class_name + ", number of subjects: " + str(num_of_subjects))
    return num_of_subjects


def get_image_id(image_name_string: str) -> str:
    substrings = image_name_string.split('_')
    full_image_id = substrings[len(substrings) - 1]  # Iid.nii
    # full_image_id[1::]  # id.nii
    image_id = full_image_id.split('.')  # id
    return image_id[0]


def get_image_id_from_filename(image_name_string: str) -> str:
    substrings = image_name_string.split('\\')
    full_image_id = substrings[len(substrings) - 1]  # Iid.nii
    # full_image_id[1::]  # id.nii
    image_id = full_image_id.split('.')  # id
    return image_id[0]


def get_image_group(image_id: str, excel_file_path: str, excel_sheet_name: str) -> str:
    xl_file = pd.ExcelFile(excel_file_path)
    dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    df = dfs[excel_sheet_name]
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
            return str(row[index_research_group_column])
