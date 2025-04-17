import pandas as pd
import ast

# Load the UK Sanctions processed dataset
sanctions_df = pd.read_csv("output/UK_Sanctions_Processed.csv")

# Load the evaluation dataset 
evaluation_df = pd.read_csv("data/evaluation_names.csv")  

# Normalize names
sanctions_df["primary_name"] = sanctions_df["primary_name"].str.lower().str.strip()

# Convert string (because CSV) representations of alias lists into actual Python lists
sanctions_df["aliases"] = sanctions_df["aliases"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
sanctions_df["aliases"] = sanctions_df["aliases"].apply(lambda alias_list: [alias.lower().strip() for alias in alias_list])

# Normalize evaluation names 
evaluation_df["name"] = evaluation_df["name"].str.lower().str.strip()

# Function to match a name against primary_name and aliases
def match_name(name, row):
    if name == row["primary_name"]:
        return True
    return name in row["aliases"]

# Search for matches
def find_matches(name):
    matches = sanctions_df[sanctions_df.apply(lambda row: match_name(name, row), axis=1)]
    return matches

# Match all names from the evaluation dataset
results = []

for idx, row in evaluation_df.iterrows():
    name = row["name"]
    matches = find_matches(name)
    if not matches.empty:
        for _, match_row in matches.iterrows():
            results.append({
                "input_name": name,
                "matched_primary_name": match_row["primary_name"],
                "group_id": match_row["group_id"],
                "match_found_in": "primary_name" if name == match_row["primary_name"] else "aliases",
                "source_row": idx
            })

# Output results
matched_df = pd.DataFrame(results)
matched_df.to_csv("output/matched_names.csv", index=False)
print("Matching completed! Results saved to: output/matched_names.csv")

