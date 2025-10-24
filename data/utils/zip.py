import os, gzip, shutil
import pandas as pd

def unparquet(input_dir:str, output_dir:str):
  if not os.path.exists(output_dir): os.makedirs(output_dir)
  parquet_files = [file for file in os.listdir(input_dir) if file.endswith('.parquet')]

  for file_name in parquet_files:
    input_path = os.path.join(input_dir, file_name)
    base_name = os.path.splitext(file_name)[0]

    # Convert to CSV
    csv_output_path = os.path.join(output_dir, base_name + '.csv')
    df = pd.read_parquet(input_path)
    df.to_csv(csv_output_path, sep=',', index=False)
    print(f"CSV Conversion complete: {file_name} -> {csv_output_path}")

    # Convert to TXT
    txt_output_path = os.path.join(output_dir, base_name + '.txt')
    df.to_csv(txt_output_path, sep='\t', index=False)
    print(f"TXT Conversion complete: {file_name} -> {txt_output_path}")
  print("All Parquet files converted to CSV and TXT files.")

def unzip(input_dir, output_dir):
  os.makedirs(output_dir, exist_ok=True)

  # List all files in the input dir
  files = os.listdir(input_dir)

    # Iterate over each file in the input dir
  for file_name in files:
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, os.path.splitext(file_name)[0])

        # Check if the file is a GZip file
    if file_name.endswith('.gz'):
      print(f"Unzipping: {file_name}")

      # Open the GZip file and decompress the data
      with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)

          print(f"Unzipping complete: {output_path}")
    else:
      print(f"Skipping non-GZip file: {file_name}")