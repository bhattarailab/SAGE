import argparse
import pandas as pd
import os
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

import imagehash

HASHING_ALG = ["average", "perceptual", "difference", "wavelet", "color", "embeddings"]

# hyperaparameter
AVERAGE_HASH_SIZE = 8
PHASH_SIZE = 16
HIGHFREQ_FACTOR = 16
DHASH_SIZE = 16
WHASH_SIZE = 16
COLOR_HASH_BIT_BITS = 16

# thresholds
AVERAGE_THRESHOLD = 2
THRESHOLD = 10
COLOR_THRESHOLD = 3
CONSINE_THRESHOLD = 0.9
WAVELET_THRESHOLD = 8

def main(df: pd.DataFrame, output_csv: str, alg: list[str], debug_path: Path | None = None):
	# create debug folder
	if debug_path is not None:
		debug_path = Path(debug_path)
		os.makedirs(debug_path, exist_ok=True)

	if "embeddings" in alg:
		embeddings = np.load("./outputs/embeddings/dinov3-b.npy")
		df["embeddings"] = embeddings.tolist()

	# initialize the value for eac hashing algorithm
	for a in alg:
		if a != "embeddings":
			df[a] = None

	# compute the hash values for each algorithm
	all_ignore_indices = []
	for row in tqdm(df.itertuples(), total=len(df),desc="Extracting Image"):
		image = Image.open(row.file_path)

		# apply all algorithms
		for a in alg:
			if a == "average":
				df.at[row.Index, a] = imagehash.average_hash(image, hash_size=AVERAGE_HASH_SIZE)
			elif a == "perceptual":
				df.at[row.Index, a] = imagehash.phash(image, hash_size=PHASH_SIZE, highfreq_factor=HIGHFREQ_FACTOR)
			elif a == "difference":
				df.at[row.Index, a] = imagehash.dhash(image, hash_size=DHASH_SIZE)
			elif a == "wavelet":
				df.at[row.Index, a] = imagehash.whash(image, hash_size=WHASH_SIZE)
			elif a == "color":
				df.at[row.Index, a] = imagehash.colorhash(image, binbits=COLOR_HASH_BIT_BITS)

	# find similar image with thresholding; distance < THREHOLD then similar
	for a in alg:
		# create debug folder for the algorithm
		if debug_path is not None:
			os.makedirs(debug_path/a, exist_ok=True)

		values = df[a].values
		if a == "embeddings":
			X = np.vstack(values)
			threshold_matrix = X @ X.T
		else:
			threshold_matrix = np.tile(values, (len(values), 1)) - values.reshape(len(values), 1)

		if a == "embeddings":
			similar_matrix = threshold_matrix > CONSINE_THRESHOLD
		elif a == "average":
			similar_matrix = threshold_matrix < AVERAGE_THRESHOLD
		elif a == "color":
			similar_matrix = threshold_matrix < COLOR_THRESHOLD
		elif a == "wavelet":
			similar_matrix = threshold_matrix < WAVELET_THRESHOLD
		else:
			similar_matrix = threshold_matrix < THRESHOLD

		ignore_indices = []
		for i, similar_mask in tqdm(enumerate(similar_matrix), total=len(similar_matrix), desc=f"Matching Similar with alg={a}: "):
			# skip this if already there
			if i in ignore_indices:
				continue

			# skip if only itself is similar
			if similar_mask.sum() <= 1:
				continue
			
			similar_indices = np.arange(len(similar_mask))[similar_mask]
			# first sample is not ignored
			ignore_indices.extend(similar_indices[similar_indices != i])

			if debug_path is not None:
				os.makedirs(debug_path/a/str(i), exist_ok=True)

				# for each create the folder
				for row in df[similar_mask].itertuples():
					Image.open(row.file_path).save(debug_path/a/str(i)/row.file_path.split(os.sep)[-1])
		
		all_ignore_indices.extend(ignore_indices)

	duplicated_removed_df = df.drop(all_ignore_indices)
	duplicated_removed_df = df.drop(columns=["findings", *alg])
	duplicated_removed_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input-csv", required=True)
	parser.add_argument("--debug-path", default=None)
	parser.add_argument("--output-csv", required=True)
	parser.add_argument("--alg", nargs="+")

	args = parser.parse_args()
	input_csv, output_csv = args.input_csv, args.output_csv

	if os.path.exists(output_csv):
		raise FileExistsError()

	df = pd.read_csv(input_csv)
	main(df, output_csv, args.alg, args.debug_path)
