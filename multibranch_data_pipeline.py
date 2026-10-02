# =========================================================
# Chapter: Multi-Branch Biomedical Data Pipeline (DataFrames)
# =========================================================
import pandas as pd
import numpy as np
import yaml
import xml.etree.ElementTree as ET
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
import matplotlib.pyplot as plt
import os

# --- 1. Synthetic Patient Data Generator ---
def compute_parameters(min_val, max_val, index):
    val = (np.random.rand() * (max_val - min_val)) + min_val
    return np.round(val) if index == 0 else np.round(val, 1)

def create_patient_data(i):
    patients = ["Chris", "Georgia", "Helen", "Mary", "George", "Mike",
                "Olivia", "Emma", "Ava", "Charlotte", "Sophia",
                "Liam", "Noah", "Oliver", "Elijah", "William"]
    name_idx = int(np.round(np.random.rand() * (len(patients) - 1)))
    
    record = (
        patients[name_idx],              # Name
        compute_parameters(20, 80, 0),   # Age
        compute_parameters(15, 40, 1),   # BMI
        compute_parameters(130, 300, 2), # Blood Sugar
        compute_parameters(100, 220, 3), # Blood Pressure
        compute_parameters(20, 80, 4),   # HDL
        compute_parameters(30, 60, 5)    # Haematocrit
    )
    return record

def create_medical_branch_file(num_patients):
    data = [create_patient_data(i) for i in range(num_patients)]
    columns = ['Name', 'Age', 'BMI', 'Bl_Sugar', 'Bl_Press', 'HDL', 'Haematocrit']
    return pd.DataFrame(data, columns=columns)

# --- 2. Centralized Pipeline Generation ---
# Simulate 4 branches generating daily reports
N = 5
branch1 = create_medical_branch_file(N)
branch2 = create_medical_branch_file(N)
branch3 = create_medical_branch_file(N)
branch4 = create_medical_branch_file(N)

# Save Branch 1 as Excel (.xlsx)
branch1.to_excel("Branch1.xlsx", header=True, index=False)

# Save Branch 2 as JSON
branch2.to_json("Branch2.json", orient="records", indent=4)

# Save Branch 3 as XML
def dict2xml_custom(dict_or_list):
    xml_bytes = dicttoxml(dict_or_list, root=True, custom_root='root', attr_type=False)
    dom = parseString(xml_bytes)
    return dom.toprettyxml()

dict_obj = branch3.to_dict(orient='records')
xml_string = dict2xml_custom(dict_obj)
with open("Branch3.xml", 'w') as file:
    file.write(xml_string)

# Save Branch 4 as YAML
with open('Branch4.yaml', 'w') as file:
    yaml.dump(branch4.to_dict(orient='records'), file, sort_keys=False)

print("Data exported successfully to XLSX, JSON, XML, and YAML.")

# --- 3. Data Ingestion & Merging ---
def read_xml_to_df(file_name):
    tree = ET.parse(file_name)
    root = tree.getroot()
    data = []
    for item in root.findall('item'):
        record = {}
        for child in item:
            record[child.tag] = child.text
        data.append(record)
    return pd.DataFrame(data)

combined_df = pd.concat([
    pd.read_excel("Branch1.xlsx"),
    pd.read_json("Branch2.json"),
    read_xml_to_df("Branch3.xml"),
    pd.DataFrame(yaml.safe_load(open('Branch4.yaml', 'r')))
], ignore_index=True)

# Convert string values to numeric
numeric_cols = ['Age', 'BMI', 'Bl_Sugar', 'Bl_Press', 'HDL', 'Haematocrit']
for col in numeric_cols:
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

# --- 4. Statistical Summary & Visualization ---
def print_statistics(df):
    print('\n--- Unified DataFrame Statistics ---')
    print("Mean:\n", df.mean())
    print("\nStandard Deviation:\n", df.std())
    print("\nSkewness:\n", df.skew())
    print("\nKurtosis:\n", df.kurtosis())

numeric_df = combined_df.select_dtypes(include=np.number).dropna()
print_statistics(numeric_df)

# Plot distributions
numeric_df.hist(figsize=(10, 8), bins=10, color='teal', edgecolor='black')
plt.suptitle('Distribution of Patient Clinical Metrics')
plt.show()
