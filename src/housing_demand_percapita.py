"""Group quarters, headship and household formation rates by NUTS 3 region, age, sex and nationality for 2011, 2021 and 2022."""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, splev, splrep


def load_and_clean_population_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, skiprows=8, encoding='latin-1')

    symbol_index = df[df['Values'] == "Symbol"].index[0]
    df = df.iloc[:symbol_index]

    df['Number'] = df['Number'].replace('-', '0')
    df['Number'] = pd.to_numeric(df['Number'], errors='coerce').fillna(0)

    df.columns = [
        'Values', 'Year', 'NUTS3', 'Sex', 'Age',
        'Nationality', 'Population', 'Annotations', 'Unnamed',
    ]

    df = df[df['NUTS3'] != 'Not classifiable <0>']

    df['Year'] = df['Year'].astype(float)

    df['NUTS3'] = df['NUTS3'].str.extract(r'<(.*?)>')

    df = df[df['Age'] != 'Not applicable']
    df['Age'] = df['Age'].replace({
        '100 plus years old': 100,
        'under 1 year old': 0,
        '1 year old': 1,
    })
    df['Age'] = df['Age'].astype(str).str.replace(' years old', '')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    df['Sex'] = df['Sex'].map({'male': 'M', 'female': 'F'})

    df = df.drop(columns=['Unnamed', 'Annotations', 'Values'], errors='ignore')
    return df


OCT_TOTAL_FILES = [
    'Data/Statcube/register_persons_2011_2021-2011-2021_age_sex_ATF_nuts.csv',
    'Data/Statcube/register_persons_2021_2024-2022_age_sex_ATF_nuts.csv',
]
OCT_PRIVATE_TEMPLATE = 'Data/Statcube/register_persons_{series}-{year}_nuts3_age_sex_private_{nat}.csv'
OCT_SERIES = {2011: '2011_2021', 2021: '2011_2021', 2022: '2021_2024'}
PRIVATE_LABELS = {'Private households', 'Persons living in a private household'}


def load_register_persons(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, skiprows=8, encoding='latin-1', dtype=str, keep_default_na=False)
    df = df.iloc[:df.index[df['Counting'] == 'Symbol'][0]]
    df = df[df['Counting'] == 'Persons']

    for c in df.columns:
        if c.startswith(('Type of household', 'Living arrangement')):
            assert set(df[c]) <= PRIVATE_LABELS, (file_path, set(df[c]))

    names = {'Year': 'Year', 'Age': 'Age', 'Sex': 'Sex', 'Citizenship': 'Nationality',
             'Place of usual residence': 'NUTS3', 'Number': 'Persons'}
    df = df.rename(columns={c: n for c in df.columns for p, n in names.items() if c.startswith(p)})
    df = df[['Year', 'NUTS3', 'Age', 'Sex', 'Nationality', 'Persons']].copy()

    df['Persons'] = pd.to_numeric(df['Persons'].replace('-', '0'))
    df['Year'] = df['Year'].astype(float)
    df['NUTS3'] = df['NUTS3'].str.extract(r'<(AT\d{3})>', expand=False)
    df['Sex'] = df['Sex'].map({'Male': 'M', 'Female': 'F'})
    df['Nationality'] = df['Nationality'].replace({
        'Not Austria (incl. Stateless/Unsettled/Unknown)': 'Foreign country',
    })
    return df


def load_october_register() -> pd.DataFrame:
    keys = ['Year', 'NUTS3', 'Age', 'Sex', 'Nationality']
    tot = pd.concat([load_register_persons(f) for f in OCT_TOTAL_FILES], ignore_index=True)
    priv = pd.concat(
        [load_register_persons(OCT_PRIVATE_TEMPLATE.format(series=s, year=y, nat=n))
         for y, s in OCT_SERIES.items() for n in ('AT', 'F')],
        ignore_index=True,
    )
    df = tot.merge(priv, on=keys, how='outer', suffixes=('_total', '_private'), validate='one_to_one')
    df = df.rename(columns={'Persons_total': 'Population', 'Persons_private': 'PrivatePopulation'})

    assert set(df['Year']) == set(map(float, OCT_SERIES)), sorted(df['Year'].unique())
    assert (df.groupby('Year').size() == 35 * 21 * 2 * 2).all()
    assert df[['NUTS3', 'Sex', 'Population', 'PrivatePopulation']].notna().all().all()
    assert (df['PrivatePopulation'] <= df['Population']).all()

    df['NPPopulation'] = df['Population'] - df['PrivatePopulation']
    return df


def rescale_population_to_october(df_pop: pd.DataFrame, df_oct: pd.DataFrame) -> pd.DataFrame:
    df = df_pop.copy()
    df['Age_Group'] = df['Age'].apply(map_age_to_group).replace({'0 to 4 years': 'Under 5 years'})
    keys = ['Year', 'NUTS3', 'Age_Group', 'Sex', 'Nationality']
    target = df_oct.rename(columns={'Age': 'Age_Group'}).set_index(keys)['Population']
    df['Target'] = target.reindex(pd.MultiIndex.from_frame(df[keys])).to_numpy()
    assert df['Target'].notna().all(), 'population cell without a 31 October count'
    assert len(df.drop_duplicates(keys)) == len(target), '31 October cell without population'

    grp = df.groupby(keys)['Population']
    jan_sum = grp.transform('sum')
    n_ages = grp.transform('size')
    df['Population'] = np.where(
        jan_sum > 0,
        df['Population'] * df['Target'] / jan_sum.where(jan_sum > 0, 1),
        df['Target'] / n_ages,
    )
    check = df.groupby(keys)['Population'].sum().reindex(target.index)
    assert np.allclose(check, target), 'rescaled groups differ from the 31 October counts'
    return df.drop(columns=['Age_Group', 'Target'])


def map_age_to_group(age) -> str:
    if pd.isna(age):
        return np.nan
    try:
        age = int(age)
    except (ValueError, TypeError):
        return np.nan

    if age < 0:
        return np.nan

    boundaries = list(range(0, 100, 5))
    for low in boundaries:
        high = low + 4
        if low <= age <= high:
            return f'{low} to {high} years'
    if age >= 100:
        return '100 years and over'
    return np.nan


def get_age_midpoint(age_group_str) -> float:
    if pd.isna(age_group_str) or not isinstance(age_group_str, str):
        return np.nan
    try:
        if 'to' in age_group_str:
            parts = age_group_str.replace(' years', '').split(' to ')
            if len(parts) != 2:
                return np.nan
            return (int(parts[0]) + int(parts[1])) / 2
        elif 'and over' in age_group_str:
            return int(age_group_str.replace(' years and over', ''))
        else:
            return np.nan
    except (ValueError, TypeError):
        return np.nan


def categorize_age(age: int) -> str:
    if age < 15:
        return 'Under 15'
    elif age <= 29:
        return '15 to 29 years'
    elif age <= 49:
        return '30 to 49 years'
    elif age <= 64:
        return '50 to 64 years'
    elif age <= 84:
        return '65 to 84 years'
    else:
        return '85 years and over'


def calculate_gqr_by_age_and_spline(
    df_np: pd.DataFrame,
    df_pop: pd.DataFrame,
    spline_ages: np.ndarray = np.arange(0, 101),
) -> pd.DataFrame:
    df_np = df_np.copy()
    df_pop = df_pop.copy()

    df_np['Year'] = pd.to_numeric(df_np['Year'], errors='coerce').astype('Int64')
    df_pop['Year'] = pd.to_numeric(df_pop['Year'], errors='coerce').astype('Int64')
    df_np = df_np.dropna(subset=['Year'])
    df_pop = df_pop.dropna(subset=['Year'])

    df_pop['Age_Group'] = df_pop['Age'].apply(map_age_to_group)
    df_pop = df_pop.dropna(subset=['Age_Group'])

    df_np = df_np.rename(columns={'Age': 'Age_Group'})
    df_np = df_np.dropna(subset=['Age_Group'])
    df_np['Age_Group'] = df_np['Age_Group'].replace({'Under 5 years': '0 to 4 years'})

    group_keys = ['Year', 'NUTS3', 'Age_Group', 'Sex', 'Nationality']

    gq_pop_summary = (
        df_np.groupby(group_keys, dropna=False)['NPPopulation']
        .sum().reset_index()
        .rename(columns={'NPPopulation': 'GQPop'})
    )
    tp_pop_summary = (
        df_pop.groupby(group_keys, dropna=False)['Population']
        .sum().reset_index()
        .rename(columns={'Population': 'TPop'})
    )

    df_gqr_by_cohort = pd.merge(
        gq_pop_summary, tp_pop_summary, on=group_keys, how='outer'
    ).fillna(0)

    df_gqr_by_cohort['GQR_by_Age'] = np.clip(
        np.where(df_gqr_by_cohort['TPop'] > 0,
                 df_gqr_by_cohort['GQPop'] / df_gqr_by_cohort['TPop'], 0),
        0, 1,
    )

    df_gqr_by_cohort['Age_Midpoint'] = df_gqr_by_cohort['Age_Group'].apply(get_age_midpoint)
    df_gqr_by_cohort = df_gqr_by_cohort.dropna(subset=['Age_Midpoint'])

    pop_lookup_by_age = (
        df_pop.set_index(['Year', 'NUTS3', 'Sex', 'Nationality', 'Age'])['Population']
        .to_dict()
    )

    splined_results = []
    spline_iter_keys = ['Year', 'NUTS3', 'Sex', 'Nationality']

    for cohort_keys, group_df in df_gqr_by_cohort.groupby(spline_iter_keys, dropna=False):
        year, nuts3, sex, nationality = cohort_keys
        group_df_sorted = group_df.sort_values(by='Age_Midpoint')
        spline_data = group_df_sorted[group_df_sorted['TPop'] > 0]

        if len(spline_data) == 0:
            continue

        x_values = spline_data['Age_Midpoint'].values
        y_values = spline_data['GQR_by_Age'].values

        _, unique_idx = np.unique(x_values, return_index=True)
        unique_idx = np.sort(unique_idx)
        x_unique = x_values[unique_idx]
        y_unique = y_values[unique_idx]

        k_degree = min(3, len(x_unique) - 1)

        if k_degree < 1:
            if len(x_unique) == 1:
                splined_gqrs = np.clip(np.full(len(spline_ages), y_unique[0]), 0, 1)
            else:
                splined_gqrs = np.zeros(len(spline_ages))
        else:
            try:
                sort_idx = np.argsort(x_unique)
                tck = splrep(x_unique[sort_idx], y_unique[sort_idx], k=k_degree)
                splined_gqrs = np.clip(splev(spline_ages, tck), 0, 1)
            except ValueError as e:
                print(
                    f"Warning: Spline interpolation failed for Year {year}, "
                    f"NUTS3 {nuts3}, Sex {sex}, Nationality {nationality} "
                    f"due to '{e}'."
                )
                print("Falling back to linear interpolation if enough points, otherwise zeros/constant.")
                if len(x_unique) >= 2:
                    splined_gqrs = np.interp(spline_ages, x_unique, y_unique)
                elif len(x_unique) == 1:
                    splined_gqrs = np.full(len(spline_ages), y_unique[0])
                else:
                    splined_gqrs = np.zeros(len(spline_ages))

        for i, age in enumerate(spline_ages):
            pop_key = (year, nuts3, sex, nationality, age)
            exact_pop = pop_lookup_by_age.get(pop_key, 0)
            gqr_val = splined_gqrs[i]
            splined_results.append({
                'Year': year,
                'NUTS3': nuts3,
                'Sex': sex,
                'Nationality': nationality,
                'Age': age,
                'GQR': gqr_val,
                'GQPop': exact_pop * gqr_val,
                'Population': exact_pop,
            })

    if not splined_results:
        return pd.DataFrame(
            columns=['Year', 'NUTS3', 'Sex', 'Nationality', 'Age', 'GQR', 'GQPop', 'Population']
        )
    return pd.DataFrame(splined_results)


def aggregate_and_calculate_headship_rate(
    df_dw: pd.DataFrame,
    df_pop: pd.DataFrame,
    grouping_vars: list[str],
) -> pd.DataFrame:
    df_dw_agg = df_dw.groupby(grouping_vars, as_index=False)['Dwellings'].sum()
    df_pop_agg = df_pop.groupby(grouping_vars, as_index=False)['HHPopulation'].sum()

    df_hs = pd.merge(df_dw_agg, df_pop_agg, on=grouping_vars, how='inner')
    df_hs['headship_rate'] = np.where(
        df_hs['HHPopulation'] == 0, 0,
        (df_hs['Dwellings'] / df_hs['HHPopulation']) * 100,
    )
    df_hs['headship_rate'] = df_hs['headship_rate'].fillna(0)
    return df_hs


def interpolate_headship_rate(
    df_hs: pd.DataFrame,
    full_age_range: np.ndarray = np.arange(15, 101),
) -> pd.DataFrame:
    age_mapping = {
        "15 to 29 years": (15, 29),
        "30 to 49 years": (30, 49),
        "50 to 64 years": (50, 64),
        "65 to 84 years": (65, 84),
        "85 years and over": (85, 100),
        "Under 15": (0, 14),
    }

    results = []
    grouping_vars = [
        col for col in df_hs.columns
        if col not in ['Age_group', 'headship_rate', 'Dwellings', 'HHPopulation']
    ]

    for keys, group in df_hs.groupby(grouping_vars):
        midpoints, rates = [], []
        for _, row in group.iterrows():
            age_group = str(row['Age_group'])
            if age_group in age_mapping:
                low, high = age_mapping[age_group]
                midpoints.append(np.mean([low, high]))
                rates.append(row['headship_rate'])

        if len(midpoints) >= 2:
            cs = CubicSpline(midpoints, rates, bc_type='natural')
            interp_rates = np.clip(cs(full_age_range), 0, 100)
        elif len(midpoints) == 1:
            interp_rates = np.full_like(full_age_range, rates[0], dtype=float)
        else:
            interp_rates = np.full_like(full_age_range, 0, dtype=float)

        if isinstance(keys, tuple):
            key_dict = dict(zip(grouping_vars, keys))
        elif len(grouping_vars) == 1:
            key_dict = {grouping_vars[0]: keys}
        else:
            key_dict = {}

        for age, rate in zip(full_age_range, interp_rates):
            row_result = key_dict.copy()
            row_result['Age'] = age
            row_result['headship_rate'] = rate
            results.append(row_result)

    return pd.DataFrame(results)


def calculate_hhfr_and_dwellings(
    df_base: pd.DataFrame,
    df_rate: pd.DataFrame,
) -> pd.DataFrame:
    value_cols = {'headship_rate', 'Dwellings', 'HHPopulation'}
    merge_keys = list(set(df_rate.columns) - value_cols)

    cols_to_merge = [c for c in merge_keys + ['headship_rate'] if c in df_rate.columns]
    if 'headship_rate' not in cols_to_merge:
        print("Warning: 'headship_rate' column not found in df_rate. Returning base df.")
        return df_base.copy()

    df_merged = df_base.merge(df_rate[cols_to_merge], on=merge_keys, how='left')

    df_merged.loc[
        (df_merged['Age'] < 15) & (df_merged['headship_rate'].isna()),
        'headship_rate',
    ] = 0
    df_merged['headship_rate'] = df_merged['headship_rate'].fillna(0)

    df_merged['HHFR'] = (1 - df_merged['GQR']) * df_merged['headship_rate']
    df_merged['Dwellings'] = df_merged['Population'] * (df_merged['HHFR'] / 100)
    return df_merged


def report_population_by_nationality_year(
    dataframe: pd.DataFrame, population_col: str, title: str,
) -> None:
    pop_by_nat_year = dataframe.groupby(['Year', 'Nationality'])[population_col].sum()
    print(f"\n--- {title} ---")
    for (year, nationality), count in pop_by_nat_year.items():
        print(f" - Year {year}, Nationality {nationality}: {count} people")


def main():
    years = [2022, 2021, 2011]
    age_groups = ['0-74', '75-100']
    sexes = ['M', 'F']
    nats = ['AT', 'F']

    pop_file_template = "Data/Statcube/population-{year}_nuts3_age{age}_{Sex}_{nat}.csv"

    df_pop = pd.DataFrame()
    for year in years:
        for age in age_groups:
            for sex in sexes:
                for nat in nats:
                    fpath = pop_file_template.format(year=year, age=age, Sex=sex, nat=nat)
                    df_pop = pd.concat(
                        [df_pop, load_and_clean_population_data(fpath)],
                        ignore_index=True,
                    )

    df_pop = df_pop.groupby(
        ['Year', 'NUTS3', 'Age', 'Sex', 'Nationality'], as_index=False
    )['Population'].sum()

    df_oct = load_october_register()
    jan_by_nat = df_pop.groupby(['Year', 'Nationality'])['Population'].sum()
    df_pop = rescale_population_to_october(df_pop, df_oct)
    oct_by_nat = df_pop.groupby(['Year', 'Nationality'])['Population'].sum()
    print('Population moved from 1 January to 31 October (census date):')
    print(pd.DataFrame({
        '1 Jan': jan_by_nat, '31 Oct': oct_by_nat,
        'change %': 100 * (oct_by_nat / jan_by_nat - 1),
    }).round(2))

    df_pop.head()
    print('2022 population after cleaning dataset ', df_pop[df_pop['Year'] == 2022]['Population'].sum())

    df_pop.to_csv('Data/Statcube/cleaned_population_data_housing.csv', index=False)

    df_dw = pd.read_csv(
        r'Data/Statcube/housing_census_dwellings-2011-2021-2022_nuts3_age_sex_ATF.csv',
        skiprows=8, encoding='latin-1',
    )
    symbol_index = df_dw[df_dw['Counting'] == "Symbol"].index[0]
    df_dw = df_dw.iloc[:symbol_index]
    df_dw.tail()

    df_dw['Number'] = df_dw['Number'].replace('-', '0')
    df_dw['Number'] = pd.to_numeric(df_dw['Number'], errors='coerce').fillna(0)
    print('Dwellings before cleaning dataset ', df_dw[df_dw['Year'] == '2011']['Number'].sum())

    df_dw.columns = [
        'Counting', 'Year', 'NUTS3', 'Age_group', 'Sex',
        'Nationality', 'Dwellings', 'Annotations', 'Unnamed',
    ]
    df_dw['Year'] = df_dw['Year'].astype(float)
    df_dw['NUTS3'] = df_dw['NUTS3'].str.extract(r'<(.*?)>')

    df_dw = df_dw[df_dw['Age_group'] != 'Not applicable']
    df_dw['Nationality'] = df_dw['Nationality'].replace({
        'Not Austria (incl. Stateless/Unsettled/Unknown)': 'Foreign country',
    })
    df_dw['Sex'] = df_dw['Sex'].map({'Male': 'M', 'Female': 'F'})

    df_dw = df_dw.groupby(
        ['Year', 'NUTS3', 'Age_group', 'Sex', 'Nationality'], as_index=False
    )['Dwellings'].sum()
    df_dw = df_dw.drop(columns=['Unnamed', 'Annotations', 'Counting'], errors='ignore')
    df_dw.head()

    df_dw['Age_group'].unique()

    print(df_dw[df_dw['Nationality'] == 'Not applicable']['Dwellings'].sum())
    df_dw = df_dw[df_dw['Nationality'] != 'Not applicable']

    dwellings_by_nat_year = df_dw.groupby(['Year', 'Nationality'])['Dwellings'].sum()
    for (year, nationality), count in dwellings_by_nat_year.items():
        print(f" - Year {year}, Nationality {nationality}: {count} dwellings")

    df_np = df_oct[['Year', 'NUTS3', 'Age', 'Sex', 'Nationality', 'NPPopulation']].copy()
    df_np.tail()

    np_pop_by_nat_year = df_np.groupby(['Year', 'Nationality'])['NPPopulation'].sum()
    for (year, nationality), count in np_pop_by_nat_year.items():
        print(f" - Year {year}, Nationality {nationality}: {count} peoople in non-private households")

    df_splined_gqr = calculate_gqr_by_age_and_spline(df_np, df_pop)
    print(df_splined_gqr)

    df_splined_gqr[df_splined_gqr['Year'] == 2022]['Population'].sum()

    df_pop_agroup = df_pop.copy()
    df_pop_agroup['Age_group'] = df_pop_agroup['Age'].apply(categorize_age)
    df_pop_agroup = df_pop_agroup.groupby(
        ['Year', 'NUTS3', 'Age_group', 'Sex', 'Nationality'], as_index=False
    )['Population'].sum()
    print("Aggregated df_pop_agroup head:")
    print(df_pop_agroup.head())

    df_np_agroup = df_np.copy()
    age_bins = [0, 15, 30, 50, 65, 85, np.inf]
    age_labels = [
        'Under 15', '15 to 29 years', '30 to 49 years',
        '50 to 64 years', '65 to 84 years', '85 years and over',
    ]
    df_np_agroup['Age_group'] = pd.cut(
        df_np_agroup['Age'].apply(
            lambda x: int(str(x).split(' ')[0]) if str(x).split(' ')[0].isdigit() else 0
        ),
        bins=age_bins, labels=age_labels, right=False, include_lowest=True,
    )
    df_np_agroup = df_np_agroup.groupby(
        ['Year', 'NUTS3', 'Age_group', 'Sex', 'Nationality'], as_index=False
    )['NPPopulation'].sum()
    print("\nAggregated df_np_agroup head:")
    print(df_np_agroup.head())

    df_hhpop = pd.merge(
        df_pop_agroup, df_np_agroup,
        on=['Year', 'NUTS3', 'Age_group', 'Sex', 'Nationality'],
        how='outer', suffixes=('_pop', '_np'),
    ).rename(columns={'Population': 'Total_population', 'NPPopulation': 'NPPopulation'})

    df_hhpop['NPPopulation'] = df_hhpop['NPPopulation'].fillna(0)
    df_hhpop['Total_population'] = df_hhpop['Total_population'].fillna(0)

    df_hhpop['HHPopulation'] = df_hhpop['Total_population'] - df_hhpop['NPPopulation']
    print("\ndf_hhpop head after merging and calculation:")
    print(df_hhpop.head())

    report_population_by_nationality_year(
        df_hhpop, 'HHPopulation',
        'Population by Nationality and Year (After Non-Private Households Adjustment)',
    )
    report_population_by_nationality_year(
        df_hhpop, 'Total_population',
        'Total Population by Nationality and Year (Before Non-Private Households Adjustment)',
    )

    aggregations = {
        'df_hs':           ['Year', 'Age_group', 'NUTS3', 'Sex', 'Nationality'],
        'df_hs_age_cob':   ['Year', 'Age_group', 'Nationality'],
        'df_hs_nuts3_cob': ['Year', 'NUTS3', 'Nationality'],
        'df_hs_sex_cob':   ['Year', 'Sex', 'Nationality'],
        'df_hs_nuts3_age': ['Year', 'NUTS3', 'Age_group'],
        'df_hs_age':       ['Year', 'Age_group'],
        'df_hs_nuts3':     ['Year', 'NUTS3'],
        'df_hs_cob':       ['Year', 'Nationality'],
        'df_hs_Sex':       ['Year', 'Sex'],
    }

    results = {}
    for key, grouping_vars in aggregations.items():
        print(f"\n--- Processing aggregation: {key} ---")
        df_agg = aggregate_and_calculate_headship_rate(
            df_dw=df_dw, df_pop=df_hhpop, grouping_vars=grouping_vars,
        )
        if 'Age_group' in grouping_vars:
            results[key] = interpolate_headship_rate(df_agg)
        else:
            results[key] = df_agg
        print(f"Sample of {key} results:")
        print(results[key].head())

    df_hs = results['df_hs']
    df_hs_age_cob = results['df_hs_age_cob']
    df_hs_nuts3_cob = results['df_hs_nuts3_cob']
    df_hs_nuts3_age = results['df_hs_nuts3_age']
    df_hs_sex_cob = results['df_hs_sex_cob']
    df_hs_age = results['df_hs_age']
    df_hs_nuts3 = results['df_hs_nuts3']
    df_hs_cob = results['df_hs_cob']
    df_hs_Sex = results['df_hs_Sex']
    print("\nAll aggregations and interpolations complete.")

    df_hs_np = df_splined_gqr[
        ['Year', 'NUTS3', 'Nationality', 'Age', 'Sex', 'Population', 'GQPop', 'GQR']
    ].merge(
        df_hs[['Year', 'NUTS3', 'Nationality', 'Age', 'Sex', 'headship_rate']],
        on=['Year', 'NUTS3', 'Nationality', 'Age', 'Sex'],
        how='left',
    )
    df_hs_np.loc[
        (df_hs_np['Age'] < 15) & (df_hs_np['headship_rate'].isna()),
        'headship_rate',
    ] = 0
    df_hs_np['Dwellings'] = (
        df_hs_np['Population']
        * (1 - df_hs_np['GQR'])
        * (df_hs_np['headship_rate'] / 100)
    )

    df_hs_age_cob_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_age_cob'],
    )
    df_hs_nuts3_cob_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_nuts3_cob'],
    )
    df_hs_sex_cob_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_sex_cob'],
    )
    df_hs_nuts3_age_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_nuts3_age'],
    )

    df_hs_age_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_age'],
    )
    df_hs_nuts3_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_nuts3'],
    )
    df_hs_cob_np = calculate_hhfr_and_dwellings(
        df_base=df_splined_gqr, df_rate=results['df_hs_cob'],
    )

    df_hs_np['HHFR'] = (1 - df_hs_np['GQR']) * df_hs_np['headship_rate']
    df_hs_np.head()

    df_hs_np.to_csv(r'XLSX/df_hs_np_clean.csv', index=False)


if __name__ == '__main__':
    main()
