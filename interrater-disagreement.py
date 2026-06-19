import krippendorff
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from sklearn.preprocessing import LabelEncoder

def normalize_boolean(x):
	if pd.isna(x) or x == "*":
		return np.nan
	
	if x in [True, "True"]:
		return 1
	
	if x in [False, "False"]:
		return 0
	
	return np.nan

table = PrettyTable()
table.field_names = ["group", "type", "alpha"]

csv_files = [
	# csv files
]
dfs = [pd.read_csv(path).sort_values(by="sample_id").reset_index(drop=True) for path in csv_files]

visibility_df = dfs[0][["sample_id", "landmark"]].rename(
    columns={"landmark": "annotator_1"}
)

for i, df in enumerate(dfs[1:], start=2):
    visibility_df = visibility_df.merge(
        df[["sample_id", "landmark"]].rename(
            columns={"landmark": f"annotator_{i}"}
        ),
        on="sample_id",
        how="outer"
    )

visibility_df.to_csv("./landmark.csv")

# validate if the sample_id is in same order across all the csv
for i in range(1, len(dfs)):
	assert all(dfs[0]["sample_id"] == dfs[i]["sample_id"])

# different measurement types
ordinal_group = ["visibility"]
nominal_group = ["section", "landmark", "instrument"] 
boolean_group = ["abnormality"]
ratio_group = ["polyp_count"]

for group, type in zip([*ordinal_group, *nominal_group, *boolean_group, *ratio_group], ["ordinal", *["nominal"]*len(nominal_group), *["boolean"]*len(boolean_group), *["ratio"]*len(ratio_group)]):
	reliability_data = [df[group].values for df in dfs]

	if type == "nominal" or type == "ordinal":
		reliability_data = [df[group].values for df in dfs]
		le = LabelEncoder()
		le.fit(np.concatenate([[v for v in row if v != "*"] for row in reliability_data]))

		reliability_data = [[le.transform([v])[0] if v != "*" else np.nan for v in row] for row in reliability_data]
	elif type == "boolean":
		reliability_data = [[normalize_boolean(rating) for rating in coder_rating] for coder_rating in reliability_data]
		type = "nominal"
	elif type == "ratio":
		reliability_data = [[int(rating) if rating != "*" else np.nan for rating in coder_rating] for coder_rating in reliability_data]
		
	alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement=type)
	table.add_row([group, type, alpha])

print(table)

breakpoint()