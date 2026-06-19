import argparse
import os
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
import json

import pandas as pd

def main(mode: str, input_csv_path: str, output_dir: Path, redact: bool):
	df = pd.read_csv(input_csv_path)
	
	# filter the df based on the mode selected
	if mode == "findings":
		df = df[~df["findings"].isna()]
	elif mode == "no-findings":
		df = df[df["findings"].isna()]

	# make output dir if not exists
	os.makedirs(output_dir, exist_ok=True)

	# for each datasample copy it from current dir to the new one
	new_df_list = []
	for row in tqdm(df.itertuples(), total=len(df), desc="Processing sample"):
			patient_uuid, visit_uuid = row.patient_uuid, row.visit_uuid
			sample_uuid = row.sample_uuid
		
			# creating the subfolders
			patient_path = output_dir/patient_uuid
			os.makedirs(patient_path, exist_ok=True)

			visit_path = patient_path/visit_uuid
			os.makedirs(visit_path, exist_ok=True)

			image_save_path = visit_path/f"{sample_uuid}.jpg"

			image = Image.open(row.file_path).convert("RGB")
			if redact:
				redact_bbox = row.redact_bbox

				if redact_bbox is None or pd.isna(redact_bbox):
					raise Exception(f"Redact bbox not found, got ({redact_bbox})")
				
				redact_bbox = json.loads(redact_bbox)
				draw = ImageDraw.Draw(image)

				for bbox in redact_bbox:
					top_left, bottom_right = bbox
					draw.rectangle([*top_left, *bottom_right], fill="black")
			
			image.save(image_save_path)

			new_df_list.append({
				"patient_uuid": patient_uuid,
				"visit_uuid": visit_uuid,
				"sample_uuid": sample_uuid,
				"file_path": str(image_save_path),
				"findings": row.findings if "findings" in row else None,
				"timestamp": row.timestamp,
			})
	
	new_df = pd.DataFrame(new_df_list)

	# no-findings donot have findings in metadata
	if mode == "no-findings":
		new_df.drop(columns=["findings"], inplace=True)

	# compute the relative timestamp for all image in seconds
	new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
	new_df["relative_timestamp"] = (new_df["timestamp"] - new_df.groupby("visit_uuid")["timestamp"].transform("min")).dt.total_seconds()

	# remove the exact timestamp (PHI Information)
	new_df.drop(columns=["timestamp"], inplace=True)
		
	new_sorted_df = new_df.sort_values(by=["patient_uuid", "visit_uuid", "relative_timestamp"])

	new_sorted_df.to_csv(output_dir/"metadata.csv", index=False)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input-csv", required=True)
	parser.add_argument("--output", required=True)
	parser.add_argument("--mode", default="both", choices=["both", "findings", "no-findings"])
	parser.add_argument("--redact", action="store_true")

	args = parser.parse_args()
	
	main(mode=args.mode, input_csv_path=args.input_csv, output_dir=Path(args.output), redact=args.redact)
