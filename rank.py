import pandas as pd

def main():
    file_path = './family_scores.csv'
    df = pd.read_csv(file_path)

    columns = df.columns
    names = df['name']

    rows = []
    for index, row in df.drop(columns=["name"]).iterrows():
        new_row = []
        for value in row:
            new_row.append((row > value).sum())
        rows.append(new_row)

    new_df = pd.DataFrame(rows, columns=columns[1:])
    new_df['name'] = names

    new_df.to_csv('family_scores_ranked.csv', index=False)

if __name__ == "__main__":
    main()