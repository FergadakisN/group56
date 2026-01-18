import pandas as pd

input_file = "./data/raw/final_all_index.txt"
output_file = "./data/processed/final_all_index.csv"

data = []
with open(input_file, "r") as file:
    for line in file:
        parts = [p.strip() for p in line.split("=")]

        if len(parts) == 5:
            data.append({
                'image_id': parts[3],
                'label': parts[1],
                'status': parts[2],
            })

df = pd.DataFrame(data)

df.to_csv(output_file, index=False)

print(f"Successfully converted {len(df)} rows to {output_file}")
print(df.head()) # Preview the first few rows
