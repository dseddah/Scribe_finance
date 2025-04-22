import pandas as pd
import json
import os
import argparse
from datetime import datetime, date
import sys
import re

def convert_cell(val):
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    return val

def sanitize_sheet_name(name):
    return re.sub(r'[^\w.-]', '_', name.strip())

def check_referenced_files(df, sheet_name, source_dir):
    """Check if files referenced in 'Document' and optionally 'Table Name (input)' exist."""
    missing_files = []

    sheet_dir = os.path.join(source_dir, sheet_name)

    for idx, row in df.iterrows():
        # Always check 'Document'
        doc_file = row.get("Document")
        if isinstance(doc_file, str) and doc_file.strip():
            doc_path = os.path.join(sheet_dir, doc_file.strip())
            if not os.path.isfile(doc_path):
                missing_files.append(doc_path)

        # Only check 'Table Name (input)' if column exists
        if "Table Name (input)" in df.columns:
            table_file = row.get("Table Name (input)")
            if isinstance(table_file, str) and table_file.strip():
                table_path = os.path.join(sheet_dir, table_file.strip())
                if not os.path.isfile(table_path):
                    missing_files.append(table_path)

    return missing_files

def main():
    parser = argparse.ArgumentParser(description="Convert Excel sheets to JSON with optional file checks.")
    parser.add_argument("--in", dest="input_file", required=True, help="Path to the input Excel file (.xlsx)")
    parser.add_argument("--out", dest="output", help="Path to output JSON file or directory. Defaults to stdout.")
    parser.add_argument("--source_data_dir", dest="source_data_dir", help="Path to source documents for validation.")
    
    args = parser.parse_args()
    input_file = args.input_file
    output_path = args.output
    source_data_dir = args.source_data_dir

    # Read all sheets
    try:
        xls = pd.read_excel(input_file, sheet_name=None, engine='openpyxl')
    except Exception as e:
        print(f"❌ Failed to read Excel file: {e}", file=sys.stderr)
        sys.exit(1)

    json_outputs = {}
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    for sheet_name, df in xls.items():
        df = df.applymap(convert_cell)

        if source_data_dir:
            missing = check_referenced_files(df, sheet_name, source_data_dir)
            for path in missing:
                print(f"⚠️  Missing file: {path}", file=sys.stderr)

        data = df.fillna('').to_dict(orient='records')
        json_outputs[sheet_name] = data

    # Output logic
    if output_path:
        if output_path.endswith('.json') and len(json_outputs) == 1:
            sheet_data = next(iter(json_outputs.values()))
            print(f"📄 Writing to {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sheet_data, f, indent=2, ensure_ascii=False)
        elif output_path.endswith('.json') and len(json_outputs) > 1:
            print("⚠️  Cannot write multiple sheets to a single JSON file. Use a directory path instead.", file=sys.stderr)
            sys.exit(1)
        else:
            os.makedirs(output_path, exist_ok=True)
            for sheet_name, data in json_outputs.items():
                safe_sheet_name = sanitize_sheet_name(sheet_name)
                filename = os.path.join(output_path, f"{base_name}.{safe_sheet_name}.json")
                print(f"📄 Writing to {filename}")
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        if len(json_outputs) == 1:
            json.dump(next(iter(json_outputs.values())), sys.stdout, indent=2, ensure_ascii=False)
        else:
            json.dump(json_outputs, sys.stdout, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
