"""Drivers of demand under doubled naturalisation.

Prints the national volume, ageing and behaviour effects in 2050, in percentage points of 2025
demand, for the main projection and the scenario of Supplementary Table 9. Writes nothing.

    python src/sensitivity_naturalisation_decomposition.py <repository root> <scenario copy>
"""

import os
import sys

import pandas as pd

import config

for label, root in [('main projection', sys.argv[1]), ('doubled naturalisation', sys.argv[2])]:
    r = pd.read_csv(os.path.join(root, config.PATH_REGIONAL_DECOMPOSITION))
    n = r[(r.Region_Code == 'NAT') & (r.Year == 2050)]
    print(label)
    for s, g in n.groupby('Service'):
        d0 = g.D0.sum()
        print('  ', s, ' | '.join(f"{row.Nationality[:7]} vol {row.Vol_Abs/d0*100:+.1f} age {row.Struct_Abs/d0*100:+.1f} beh {row.Beh_Abs/d0*100:+.1f}" for row in g.itertuples()))
