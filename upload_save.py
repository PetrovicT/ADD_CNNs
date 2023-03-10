import pandas as pd
pd.set_option('display.max_rows', None)


def upload_csv_number_of_subjects(csv_path, class_name):
    table = pd.read_csv(csv_path)
    unique_subjects = []
    for subject in range(len(table['Subject ID'])):
        if (table['Subject ID'][subject] not in unique_subjects) and table["Research Group"][subject] == class_name:
            unique_subjects.append(table['Subject ID'][subject])
    print("File: " + csv_path)
    print("Class: " + class_name + ", number of subjects: " + str(len(unique_subjects)))


def upload_excel_number_of_subject(excel_path, class_name):
    xl_file = pd.ExcelFile(excel_path)
    dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    df = dfs["tableOR_original"]
    columns_to_take = [
        "Subject ID",
        "Sex",
        "Research Group",
        "Archive Date",
        "Age",
        "Description"
    ]
    df = df.loc[:, columns_to_take]
    # print(df.shape)  # print rows, cols

    unique_subjects = []
    index_research_group_column = 0
    index_subject_id = 0
    for index in range(len(columns_to_take)):
        if columns_to_take[index] == "Research Group":
            index_research_group_column = index
        if columns_to_take[index] == "Subject ID":
            index_subject_id = index

    for row in df.to_numpy():
        if row[index_subject_id] not in unique_subjects and row[index_research_group_column] == class_name:
            unique_subjects.append(row[index_subject_id])

    print("File: " + excel_path)
    print("Class: " + class_name + ", number of subjects: " + str(len(unique_subjects)))


def split_string_find_image_id(image_name_string):
    substrings = image_name.split("_")
    image_id_length = len(substrings)
    full_image_id = substrings[image_id_length - 1]  # Iid
    return full_image_id[1::]  # returns only id, without letter I


if __name__ == '__main__':
    # path_csv = './csv_files/tableOR_original.csv'
    # upload_csv_number_of_subjects(path_csv, "AD")
    # upload_csv_number_of_subjects(path_csv, "CN")
    path_excel = './csv_files/tableOR_original.xlsx'
    upload_excel_number_of_subject(path_excel, "AD")
    upload_excel_number_of_subject(path_excel, "CN")

    image_name = "ADNI_002_S_0295_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070319114336780_S13407_I45112"
    print(split_string_find_image_id(image_name))
