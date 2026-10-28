"""Start populations from the STATcube tables: residents on 1 January 2022 and 2025, and foreign residents on 1 January 2024 and 2025, by NUTS 3 region, sex and single age."""

import pandas as pd

import config

PATH_TOTALS_2022 = r'XLSX/AT_population_totals_2022.csv.gz'
PATH_TOTALS_2025 = config.PATH_POPULATION_2025
PATH_FOREIGN_2024 = config.PATH_FOREIGN_POPULATION_2024
PATH_FOREIGN_2025 = config.PATH_FOREIGN_POPULATION_2025


def _load_statcube_population(file_path, columns):
    df = pd.read_csv(file_path, skiprows=8, encoding='latin-1')
    symbol_index = df[df['Values'] == "Symbol"].index[0]
    df = df.iloc[:symbol_index]

    df['Number'] = df['Number'].replace('-', '0')
    df['Number'] = pd.to_numeric(df['Number'], errors='coerce').fillna(0)

    df.columns = columns

    df = df[df['geo'] != 'Not classifiable <0>']

    df['year'] = df['year'].astype(float)
    df['year'] = df['year']

    df['geo'] = df['geo'].str.extract(r'<(.*?)>')

    df = df[df['age'] != 'Not applicable']

    df['age'] = df['age'].replace({
        '100 plus years old': 100,
        'under 1 year old': 0,
        '1 year old': 1
    })
    df['age'] = df['age'].astype(str).str.replace(' years old', '')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    df['sex'] = df['sex'].map({
        'male': 'M',
        'female': 'F'
    })

    df = df.drop(columns=['Unnamed', 'Annotations', 'Values'], errors='ignore')

    return df


def load_population_2022(file_path):
    return _load_statcube_population(
        file_path,
        ['Values', 'year', 'geo', 'sex', 'age', 'nationality', 'population', 'Annotations', 'Unnamed'])


def load_population_by_age_first(file_path):
    return _load_statcube_population(
        file_path,
        ['Values', 'year', 'geo', 'age', 'nationality', 'sex', 'population', 'Annotations', 'Unnamed'])


def load_population_2025(file_path):
    return _load_statcube_population(
        file_path,
        ['Values', 'year', 'geo', 'age', 'sex', 'nationality', 'population', 'Annotations', 'Unnamed'])


def convert_age_str(age_str):
    if age_str == 'Y_LT1':
        return 0
    if age_str == 'Y_GE100':
        return 100
    return int(age_str[1:])


def convert_age_to_str(age):
    if age == 0:
        return 'Y_LT1'
    if age >= 100:
        return 'Y_GE100'
    return f'Y{age}'


def build_foreign_population(df_pop_foreign, year):
    df = (
        df_pop_foreign
        .assign(age_num=lambda d: d['age_num'].astype(int))
        .groupby(['geo', 'sex', 'age_num'], as_index=False)['population']
        .sum()
        .rename(columns={'population': 'OBS_VALUE'})
    )
    df['age'] = df['age_num'].apply(convert_age_to_str)
    df['TIME_PERIOD'] = year
    df['projection'] = 'BSL'
    df['OBS_VALUE'] = df['OBS_VALUE'].round().astype(int)
    return df[['TIME_PERIOD', 'geo', 'sex', 'age', 'projection', 'OBS_VALUE', 'age_num']]


def build_totals(df_pop):
    df_tot = df_pop.groupby(['year', 'geo', 'age_num', 'sex'])['population'].sum().reset_index()

    df_tot['age'] = df_tot['age_num'].apply(convert_age_to_str)

    df_tot = df_tot.rename(columns={'year': 'TIME_PERIOD', 'population': 'OBS_VALUE'})
    df_tot = df_tot.groupby(['geo', 'TIME_PERIOD', 'age', 'OBS_VALUE', 'sex']).sum().reset_index()
    df_tot['projection'] = 'BSL'
    return df_tot


def load_statcube_totals(year, loader):
    file_template = "Data/Statcube/population-{year}_nuts3_age{age}_{sex}_{nat}.csv"

    df_pop = pd.DataFrame()

    ages = ['0-74', '75-100']
    sexes = ['M', 'F']
    nationalities = ['AT', 'F']

    for age in ages:
        for sex in sexes:
            for nat in nationalities:
                file_path = file_template.format(year=year, age=age, sex=sex, nat=nat)
                df_cleaned = loader(file_path)
                df_pop = pd.concat([df_pop, df_cleaned], ignore_index=True)

    df_pop = df_pop.rename(columns={'age': 'age_num'})
    return df_pop


def make_totals_2022():
    df_pop = load_statcube_totals('2022', load_population_2022)
    df_tot_2022 = build_totals(df_pop)
    print(f"Saving {PATH_TOTALS_2022}")
    df_tot_2022.to_csv(PATH_TOTALS_2022, index=False)


def make_foreign_pop_2024():
    file_template = "Data/Statcube/population-2024_nuts3_age_{type}_sex_foreign.csv"

    df_pop = pd.DataFrame()

    ages = ['0-54', '55-100']

    for age in ages:
        file_path = file_template.format(type=age)
        df_cleaned = load_population_by_age_first(file_path)
        df_pop = pd.concat([df_pop, df_cleaned], ignore_index=True)

    df_pop = df_pop.rename(columns={'age': 'age_num'})

    result = build_foreign_population(df_pop, 2024)
    print(f"Saving {PATH_FOREIGN_2024} ({result['OBS_VALUE'].sum():,} persons)")
    result.to_csv(PATH_FOREIGN_2024, index=False)


def make_2025():
    df_pop = load_statcube_totals('2025', load_population_2025)

    df_pop_foreign = df_pop[df_pop['nationality'] == 'Foreign country']
    result = build_foreign_population(df_pop_foreign, 2025)
    print(f"Saving {PATH_FOREIGN_2025} ({result['OBS_VALUE'].sum():,} persons)")
    result.to_csv(PATH_FOREIGN_2025, index=False)

    df_tot = build_totals(df_pop)
    print(f"Saving {PATH_TOTALS_2025}")
    df_tot.to_csv(PATH_TOTALS_2025, index=False)


def main():
    make_totals_2022()
    make_foreign_pop_2024()
    make_2025()
    print("Done.")


if __name__ == "__main__":
    main()
