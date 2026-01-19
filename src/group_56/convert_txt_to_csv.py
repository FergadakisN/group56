import pandas as pd


def parse_index_file(input_file: str) -> pd.DataFrame:
    data = []
    with open(input_file) as file:
        for line in file:
            parts = [p.strip() for p in line.split("=")]

            if len(parts) == 5:
                data.append(
                    {
                        "image_id": parts[3],
                        "label": parts[1],
                        "status": parts[2],  # controlled/in-situ
                    }
                )

    return pd.DataFrame(data)


if __name__ == "__main__":
    input_file = "data/raw/final_all_index.txt"
    output_file = "data/processed/metadata.csv"

    df = parse_index_file(input_file)
    df.to_csv(output_file, index=False)
    print(f"Successfully converted {len(df)} rows to {output_file}")
    # print(df.head()) # Preview the first few rows
