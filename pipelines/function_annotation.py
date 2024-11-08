import pandas as pd
import json

# load data
with open('../data/query_all_heme_vars.json', 'r', encoding='latin-1') as file:
    metadata = json.load(file)

# find all functional assay types
print("Finding assay types...")
# find all functional assay types
func_types = [j[0] for i in metadata for j in i["functionalStudies"]]
func_types = list(set(func_types))
func_annotations = {x: [] for x in func_types}
func_annotations["name"] = []

print(func_annotations)

# create separate columns for each assay type
print("Extracting functional annotations...")
for x in metadata:
    name = x["name"]
    func_obj = x["functionalStudies"]
    local_annotation = {y: None for y in func_types}
    local_annotation["name"] = name

    for z in func_obj:
        local_annotation[z[0]] = z[-1]

    for assay, annotation in local_annotation.items():
        func_annotations[assay].append(annotation)

    # counting for sanity
    if len(func_annotations["Assays not specified"])%100 == 0:
        print(f"Processed {len(func_annotations['Assays not specified'])} variants")
    else:
        continue

func_columns = pd.DataFrame(func_annotations)
print("Results:")
print(func_columns.head())
print("===== Merging with embeddings =====")
data = pd.read_csv('../data/hbvar_with_embeddings.csv')
print(data.head())
data.drop(columns=['functionalStudies'], inplace=True)
data = pd.merge(data, func_columns, how='left', left_on="name", right_on="name")
print("Results:")
print(data.head())
print("===== Saving to file =====")
path = '../data/hbvar-w-func-embed-seq.csv'
data.to_csv(path, index=False)
print(f"Saved to file: {path}")