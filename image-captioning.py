import os
import json
import openai
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from dotenv import load_dotenv

from prompts.captioning import get_image_captioning_system_prompt

# load the environment variables
load_dotenv()

def add_entry_to_checkpoint(id: str, checkpoint_path, lock):
	with lock:
		with open(checkpoint_path, "r") as fp:
			data = json.load(fp)
		
		data.append(id)

		with open(checkpoint_path, "w") as fp:
			json.dump(data, fp)

def image2caption(client, model, sample_id, file_path, output_path, checkpoint_path, lock):
	with open(file_path, "rb") as file:
		base64_image = base64.b64encode(file.read()).decode("utf-8")
	
	response = client.chat.completions.create(
		model=model,
		temperature=0,
		messages=[
			{
				"role": "user",
				"content": [
					{
						"type": "text", 
						"text": get_image_captioning_system_prompt()
					},
					{
						"type": "image_url",
						"image_url": f"data:image/jpeg;base64,{base64_image}"
					}
				],
			},
		]
	)

	content = response.choices[0].message.content.strip()

	with open(output_path/f"{sample_id}.txt", "w") as fp:
		fp.write(content)

	add_entry_to_checkpoint(sample_id, checkpoint_path, lock)

def main(input_data, model, output_path, checkpoint_path):
	lock = Lock()
	client = openai.OpenAI(
		api_key=os.getenv("OPENROUTER_KEY"),
		base_url="https://openrouter.ai/api/v1",
	)

	def task(item):
		sample_id, file_path = item
		return image2caption(client, model, sample_id, file_path, output_path, checkpoint_path, lock)


	with ThreadPoolExecutor(max_workers=4) as executor:
		futures = [executor.submit(task, item) for item in input_data.items()]

		for future in tqdm(as_completed(futures), total=len(futures), desc="Processing_samples:"):
			future.result()

if __name__ == "__main__":
	parser = ArgumentParser(description="Extracts VQA and metadata for each questions given the description")

	parser.add_argument("--model", required=True)
	parser.add_argument("--name", required=True)
	parser.add_argument("--gut-vlm", action="store_true")

	args = parser.parse_args()
	
	# open the dataset
	if not args.gut_vlm:
		df = pd.read_csv("./outputs/complete-dataset.csv")
		test_df = df[df["split"] == "test"]
		print(f"Loaded data with {len(test_df)} samples.")
		
		input_data = test_df[["sample_uuid", "file_path"]].to_dict(orient="records")
		input_data = {d["sample_uuid"]: d["file_path"] for d in input_data}
	else:
		with open("./outputs/gutvlm-test.json", "r") as fp:
			data = json.load(fp)
		
		input_data = {d[:-4]:f"./outputs/kvasir-dataset-v2/{d}" for d in data.keys()}

	# check if all the image path are true
	for file_path in input_data.values():
		if not os.path.exists(file_path):
			raise FileNotFoundError(f"File not found, {file_path}")
		
	base_path = Path("./outputs/captions")
	output_path = base_path / args.name
	checkpoint_path = output_path / "checkpoint.json"
	
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	# if the plan is to restart from the previous checkpoint
	if os.path.exists(checkpoint_path):
		with open(checkpoint_path, "r") as fp:
			checkpoint_ids = json.load(fp)

		print(f"Checkpoint found with {len(checkpoint_ids)} already completed samples.")
		
		# remove those descriptions whose id is present in checkpoint
		input_data = {sample_id: file_path for sample_id, file_path in input_data.items() if sample_id not in checkpoint_ids}
	else:
		with open(checkpoint_path, "w") as fp:
			json.dump([], fp)
	
	main(input_data, args.model, output_path, checkpoint_path)
	
	# prepare the json file
	output_descriptions = {}
	with open(checkpoint_path, "r") as fp:
		all_sample_ids = json.load(fp)

	for sample_uuid in all_sample_ids:
		with open(output_path/f"{sample_uuid}.txt", "r") as fp:
			description = fp.read()

		output_descriptions[sample_uuid] = description

	with open(f"./outputs/descriptions/{args.name}.json", "w") as fp:
		json.dump(output_descriptions, fp, indent=2)
	