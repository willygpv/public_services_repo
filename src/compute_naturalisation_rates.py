"""Naturalisation rates by region, sex and age: mean resident naturalisations 2022-2023 over the 2024 foreign population, with the NUTS 2 rates applied to every NUTS 3 region and single age."""

import pandas as pd
import os
import sys

import config


PATH_NAT_DATA = r'Data/Statcube/naturalisations-2013-2023_nuts2_sex_age_groups.csv'
PATH_POP_DATA = config.PATH_FOREIGN_POPULATION_2024
PATH_OUTPUT   = config.PATH_NATURALISATION

NAT_BIN_MAP = {
    'up to 14 years old':   list(range(0, 15)),
    '15 to 17 years old':   list(range(15, 18)),
    '18 to 24 years old':   list(range(18, 25)),
    '25 to 34 years old':   list(range(25, 35)),
    '35 to 44 years old':   list(range(35, 45)),
    '45 to 54 years old':   list(range(45, 55)),
    '55 to 59 years old':   list(range(55, 60)),
    '60 to 64 years old':   list(range(60, 65)),
    '65 to 74 years old':   list(range(65, 75)),
    '75 plus years old':    list(range(75, 101))
}

NUTS2_NAME_TO_CODE = {
    'Burgenland': 'AT11', 'Carinthia': 'AT21', 'Lower Austria': 'AT12',
    'Upper Austria': 'AT31', 'Salzburg': 'AT32', 'Styria': 'AT22',
    'Tyrol': 'AT33', 'Vorarlberg': 'AT34', 'Vienna': 'AT13'
}


def parse_age(age_str):
    if age_str == 'Y_LT1': return 0
    if age_str == 'Y_GE100': return 100
    try:
        return int(str(age_str).replace('Y', '').replace('_', ''))
    except ValueError:
        return None

def get_age_to_bin_map():
    age_to_bin = {}
    for label, ages in NAT_BIN_MAP.items():
        for age in ages:
            age_to_bin[age] = label
    return age_to_bin


def load_and_clean_naturalisation_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    df = pd.read_csv(file_path, skiprows=8, encoding='latin-1')

    if "Symbol" in df['Values'].values:
        symbol_index = df[df['Values'] == "Symbol"].index[0]
        df = df.iloc[:symbol_index]

    df['Number'] = df['Number'].replace('-', '0')
    df['Number'] = pd.to_numeric(df['Number'], errors='coerce').fillna(0)

    df.columns = ['Values', 'TIME_PERIOD', 'age', 'NUTS2', 'sex', 'naturalisation', 'annotations', 'unnamed']

    df = df[df['NUTS2'] != 'Not classifiable <0>']
    df = df[df['NUTS2'] != 'Foreign country']
    df = df[df['age'] != 'Not applicable']

    df['TIME_PERIOD'] = df['TIME_PERIOD'].astype(float)
    df['sex'] = df['sex'].replace({'male': 'M', 'female': 'F'})

    df = df.drop(columns=['unnamed', 'annotations', 'Values'], errors='ignore')

    return df

def load_population_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    df = pd.read_csv(file_path)
    return df


def calculate_block_rates(df_nat, df_pop):
    age_to_bin_label = get_age_to_bin_map()

    df_pop['age_num'] = df_pop['age'].apply(parse_age)
    df_pop['NUTS2'] = df_pop['geo'].str[:4]
    df_pop['population'] = df_pop['OBS_VALUE']

    df_pop['bin_label'] = df_pop['age_num'].map(age_to_bin_label)

    pop2_binned = (
        df_pop
        .groupby(['NUTS2', 'sex', 'bin_label'], as_index=False)['population']
        .sum()
        .rename(columns={'population': 'pop_bin_total'})
    )

    if df_nat['NUTS2'].str.len().max() > 4:
        df_nat['NUTS2'] = df_nat['NUTS2'].map(NUTS2_NAME_TO_CODE)

    df_nat_avg = (
        df_nat[df_nat['TIME_PERIOD'].isin([2022, 2023, 2024])]
        .groupby(['NUTS2', 'sex', 'age'], as_index=False)['naturalisation']
        .mean()
        .rename(columns={'naturalisation': 'avg_nats_bin', 'age': 'bin_label'})
    )

    df_rates = pd.merge(
        df_nat_avg,
        pop2_binned,
        on=['NUTS2', 'sex', 'bin_label'],
        how='inner'
    )

    df_rates['block_rate'] = df_rates['avg_nats_bin'] / df_rates['pop_bin_total']
    df_rates['block_rate'] = df_rates['block_rate'].fillna(0)

    expanded_rows = []
    for _, row in df_rates.iterrows():
        bin_lbl = row['bin_label']
        rate = row['block_rate']
        nuts2 = row['NUTS2']
        sex = row['sex']

        target_ages = NAT_BIN_MAP.get(bin_lbl, [])

        for age in target_ages:
            expanded_rows.append({
                'NUTS2': nuts2,
                'sex': sex,
                'age_num': age,
                'base_nat_rate': rate
            })

    return pd.DataFrame(expanded_rows), df_pop


def main():
    print("Starting naturalisation rate redistribution...")

    print(f"Loading naturalisation data from: {PATH_NAT_DATA}")
    df_nat = load_and_clean_naturalisation_data(PATH_NAT_DATA)

    print(f"Loading population data from: {PATH_POP_DATA}")
    df_pop_raw = load_population_data(PATH_POP_DATA)

    print("Calculating block rates and broadcasting to single ages...")
    df_smooth_rates, df_pop_processed = calculate_block_rates(df_nat, df_pop_raw)

    final_nat = df_pop_processed.copy()

    final_nat = pd.merge(
        final_nat,
        df_smooth_rates,
        on=['NUTS2', 'sex', 'age_num'],
        how='left'
    )

    final_nat['base_nat_rate'] = final_nat['base_nat_rate'].fillna(0)

    final_nat['nat_rate'] = final_nat['base_nat_rate']

    print(f"Exporting results to: {PATH_OUTPUT}")
    final_nat.to_csv(PATH_OUTPUT, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
