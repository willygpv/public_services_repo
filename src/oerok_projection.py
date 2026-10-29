"""OEROK 2021-based population and household projections by NUTS 3 region."""

import pandas as pd

import config

BIRTH_GROUP_TO_NATIONALITY = {1: 'Austria', 2: 'Foreign country'}
SEX_CODES = {1: 'M', 2: 'F'}
N_NUTS3 = 35


def load_oerok_population(years):
    raw = pd.read_csv(config.PATH_OEROK_POPULATION, sep=';', decimal=',', encoding='latin-1')
    raw.columns = ['Prognoseregion', 'BirthGroup', 'SexCode', 'Age'] + list(raw.columns[4:])

    codes = pd.read_excel(config.PATH_OEROK_REGION_CODES, sheet_name='Prognoseregionen')
    region_to_nuts3 = dict(zip(
        codes['Unnamed: 0'], 'AT' + codes['NUTS-3 Region'].astype(int).astype(str),
    ))
    raw['NUTS3'] = raw['Prognoseregion'].map(region_to_nuts3)
    raw['Sex'] = raw['SexCode'].map(SEX_CODES)
    raw['Nationality'] = raw['BirthGroup'].map(BIRTH_GROUP_TO_NATIONALITY)
    if raw[['NUTS3', 'Sex', 'Nationality']].isna().any().any():
        raise ValueError("Unmapped region, sex or country-of-birth code in the ÖROK population file")
    if raw['NUTS3'].nunique() != N_NUTS3:
        raise ValueError(f"ÖROK population covers {raw['NUTS3'].nunique()} NUTS 3 regions, not {N_NUTS3}")

    year_cols = [str(y) for y in years]
    population = (
        raw.groupby(['NUTS3', 'Age', 'Sex', 'Nationality'])[year_cols].sum()
        .reset_index()
        .melt(id_vars=['NUTS3', 'Age', 'Sex', 'Nationality'],
              var_name='Year', value_name='Population')
    )
    population['Year'] = population['Year'].astype(int)
    return population


def load_oerok_households(years):
    raw = pd.read_excel(config.PATH_OEROK_HOUSEHOLDS, sheet_name='Anzahl der Haushalte', header=None)
    table_years = [int(v) for v in raw.iloc[3, 2:]]
    start = raw.index[raw.iloc[:, 0].astype(str).str.startswith('NUTS 3')][0]
    block = raw.iloc[start + 1:start + 1 + N_NUTS3, 1:2 + len(table_years)].copy()
    block.columns = ['Region'] + table_years

    names = pd.read_excel(config.PATH_OEROK_REGION_CODES, sheet_name='Sheet1')
    name_to_nuts3 = dict(zip(names['Name'].str.strip(), names['Code']))
    block['NUTS3'] = block['Region'].astype(str).str.strip().map(name_to_nuts3)
    if block['NUTS3'].isna().any() or block['NUTS3'].nunique() != N_NUTS3:
        missing = block.loc[block['NUTS3'].isna(), 'Region'].tolist()
        raise ValueError(f"ÖROK household table: NUTS 3 names not matched: {missing}")

    households = block.melt(id_vars=['NUTS3'], value_vars=list(years),
                            var_name='Year', value_name='Households')
    households['Year'] = households['Year'].astype(int)
    households['Households'] = households['Households'].astype(float)
    return households[['Year', 'NUTS3', 'Households']]


def load_oerok_reference(years):
    return {
        'population': load_oerok_population(years),
        'households': load_oerok_households(years),
    }
