import os
import pandas as pd
import numpy as np
import dask.dataframe as dd

from astropy.table import Table

DATA_DIR = "/home/kryo6156/work/data/XP_metallicity_catalogues"

#Load in the relevant catalogues (Yao GP, GC, GT, Andrae, Pristine)
#Yao et al. 2024
df_GP = pd.read_csv(os.path.join(DATA_DIR, "yao_2024/Classifier_GP.csv"))
print("yao_GP_loaded")
df_GC = pd.read_csv(os.path.join(DATA_DIR, "yao_2024/Classifier_GC.csv"))
print("yao_GC_loaded")
df_T = pd.read_csv(os.path.join(DATA_DIR, "yao_2024/Classifier_T.csv"))
print("yao_T_loaded")

#Andrae, Rix, Chandra 2023
dat = Table.read(os.path.join(DATA_DIR, "andrae_2023/table_2_catwise.fits"), 
                    format='fits')
df_and = dat.to_pandas()
print("Andrae loaded")

#Dask data fram for Pristine data, to be converted to pandas soon
dd_pris = dd.read_csv("/home/kryo6156/work/projects/metpoor_substructure/"\
			+ "data_tables/Pristine/"\
			+ "Pristine_Gaia_synthetic_FeH_v0.9_2023_08_01.csv")

print("Pris loaded")

dec_lim = 5

# Cut all tables down to G(BP) < 16 and DEC < 5 and other quality cuts

#Yao high purity giant sample (GP)
df_GP_proc = df_GP.loc[(df_GP.BP < 16) & (df_GP.DEC < dec_lim), 
			["source_id", "RA", "DEC", "BP", "p0"]]

#Yao high completeness giant sample (GC)
df_GC_proc = df_GC.loc[(df_GC.BP < 16) & (df_GC.DEC < dec_lim) & 
				(df_GC.p0 > 0.9), 
			["source_id", "RA", "DEC", "BP", "p0"]]

df_T_proc = df_T.loc[(df_T.BP < 16) & (df_T.DEC < dec_lim), 
			["source_id", "RA", "DEC", "BP", "p0"]]

df_and_proc = df_and.loc[(df_and.phot_g_mean_mag < 16) & 
				(df_and.dec < dec_lim) &
				(df_and.mh_xgboost < -2),
			["source_id", "ra", "dec", "phot_g_mean_mag", 
				"mh_xgboost", "teff_xgboost", "logg_xgboost"]]

print("processing pristine")

# Apply cuts to Pristine data, as described in Martin et al. 2023, Section 7.3

df_pris_proc = dd_pris.loc[
#                    (dd_pris.mcfrac_CaHKsyn > 0.8) & 
#                   (0.5*(dd_pris.FeH_CaHKsyn_84th-dd_pris.FeH_CaHKsyn_16th) 
#                                                                    < 0.5) &
#                   (0.5*dd_pris['E(B-V)'] < 0.5) &
#                   (0.5*(dd_pris.FeH_84-dd_pris.FeH_16) < 0.5) &
                   (dd_pris.FeH_CaHKsyn < -2) &
#                   (np.abs(dd_pris.Cstar < 3*dd_pris.Cstar_1sigma)) &
		   (dd_pris.G_0 < 16) &
		   (dd_pris.Dec < dec_lim), 
#		   (dd_pris.FeH_CaHKsyn_84th > -3.999),
                   ["source_id", "RA", "Dec", "G_0", "FeH_CaHKsyn"]].compute()

print("done processing pristine")

print("df_yao_GP %d" %len(df_GP_proc))
print("df_yao_GC %d" %len(df_GC_proc))
print("df_yao_T %d" %len(df_T_proc))
print("df_andrae %d" %len(df_and_proc))
print("df_pris %d" %len(df_pris_proc))

df_and_proc.rename(columns= {"ra": "RA", "dec": "DEC", 
                             "phot_g_mean_mag": "G_0"},
			inplace=True)

# Combine all tables into one
# 1. all stars from Yao GP, 
# 2. stars from Yao GC with P0 > 0.9 and in Pristine and not in Andrae. 
# 3. all stars in Yao T
# 4. all stars from Andrae with [M/H] < -2 and not in Yao GP

print(len(df_GC_proc[df_GC_proc.source_id.isin(df_pris_proc.source_id)]))
print(len(df_T_proc))
print(len(df_GP_proc))
print(len(df_and_proc[~df_and_proc.source_id.isin(df_GP_proc.source_id)]))

combined_df = pd.concat(
                [df_GP_proc[~df_GP_proc.source_id.isin(df_and_proc.source_id)], 
		 df_GC_proc[df_GC_proc.source_id.isin(df_pris_proc.source_id) &
			    ~df_GC_proc.source_id.isin(df_and_proc.source_id)],
		df_T_proc,
		df_and_proc])
#		df_and_proc[~df_and_proc.source_id.isin(df_GP_proc.source_id)]])
#		df_and_proc])

print(len(combined_df))

df_nodups = combined_df.drop_duplicates(subset=["source_id"])

print(len(df_nodups))

print(df_nodups.head())
#Save table
df_nodups.to_csv("../data_tables/S4_new_catalogue_13_07_24_no_dups.csv", 
                        index=False)
