#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: dataframe-2-pandera-schema-yaml.py
#------------------------------------------------------------------------------
# Version 0.1
# 9 December, 2023
# Michael Taylor
# https://patternizer.github.io
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pandera as pa
import pickle
import yaml

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

filetype = 'temp'   # ['temp', 'anom', 'normals', 'sds']
use_lek = True      # [ True, False ]

if use_lek == False: 	   
    
	df_temp_file = '../glosat-py/OUT/1781-2022-input/GloSAT.prelim04c.EBC/df_temp_qc.pkl'
	df_anom_file = '../glosat-py/OUT/1781-2022-input/GloSAT.prelim04c.EBC/df_anom_qc.pkl'
	df_normals_file = '../glosat-py/OUT/1781-2022-input/GloSAT.prelim04c.EBC/df_normals_qc.pkl'
	df_sds_file = '../glosat-py/OUT/1781-2022-input/GloSAT.prelim04c.EBC/df_sds_qc.pkl'
	glosat_version = 'GloSAT.p04c.EBC'

else:

	df_temp_file = '../glosat-py/OUT/1781-2022-input/GloSAT.prelim04c.EBCv0.6.LEKnorms21Nov22/df_temp_qc.pkl'
	df_anom_file = '../glosat-py/OUT/1781-2022-input/GloSAT.prelim04c.EBCv0.6.LEKnorms21Nov22/df_anom_qc.pkl'
	df_normals_file = '../glosat-py/OUT/1781-2022-input/GloSAT.prelim04c.EBCv0.6.LEKnorms21Nov22/df_normals_qc.pkl'
	df_sds_file = '../glosat-py/OUT/1781-2022-input/GloSAT.prelim04c.EBCv0.6.LEKnorms21Nov22/df_sds_qc.pkl'
	glosat_version = 'GloSAT.p04c.EBCv0.6.LEKnorms21Nov22'
    
if filetype == 'temp':
    
	df_file = df_temp_file
	yaml_file = 'df_temp.yml'
	json_file = 'df_temp.json'

elif filetype == 'anom':

	df_file = df_anom_file
	yaml_file = 'df_anom.yml'
	json_file = 'df_anom.json'

elif filetype == 'normals':

	df_file = df_normals_file
	yaml_file = 'df_normals.yml'
	json_file = 'df_normals.json'

elif filetype == 'sds':

	df_file = df_sds_file
	yaml_file = 'df_sds.yml'
	json_file = 'df_sds.json'

#------------------------------------------------------------------------------
# LOAD: pickled dataframe
#------------------------------------------------------------------------------

df = pd.read_pickle( df_file, compression='bz2' )

#------------------------------------------------------------------------------
# INFER: pandera data schema
#------------------------------------------------------------------------------

schema = pa.infer_schema(df)

#------------------------------------------------------------------------------
# EXPORT: to .yaml
#------------------------------------------------------------------------------

schema.to_yaml( yaml_file )

#------------------------------------------------------------------------------
# EXPORT: to .json
#------------------------------------------------------------------------------

schema.to_json( json_file, indent=4 )

#------------------------------------------------------------------------------
print('** END')
