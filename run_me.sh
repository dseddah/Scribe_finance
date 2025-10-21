#!/bin/sh

# 1st version
#python excel_to_json.py --in Q\&A_finance_and_others\ \(beta\ version\).20250422-1601.xlsx --out dataset_json --source_data_dir raw_documents/


# 2snd version
#python excel_to_json.py --in Q\&A_finance_and_others\ \(beta\ version\).20251020-2141.xlsx --out dataset_json_20251020-2141 --source_data_dir raw_documents/


# 3rd version (avec id)

FILEIN='Q\&A_finance_and_others\ \(beta\ version\).20251021-1055.xlsx'
#ls $FILEIN
python excel_to_json.py --in Q\&A_finance_and_others\ \(beta\ version\).20251021-1055.xlsx  --out dataset_json_20251021-1055 --source_data_dir raw_documents/

