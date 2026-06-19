import os
import json
import openai
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv

from prompts.extract_vqa import get_vqa_extraction_prompt

# load the environment variables
load_dotenv()

def add_entry_to_checkpoint(id: str, checkpoint_path):
	with open(checkpoint_path, "r") as fp:
		data = json.load(fp)
	
	data.append(id)

	with open(checkpoint_path, "w") as fp:
		json.dump(data, fp)

def main(descriptions, output_path, checkpoint_path, mini=False):
	client = openai.OpenAI(
		api_key=os.getenv("OPENAI_API_KEY"),
		base_url=os.getenv("BASE_URL"),
	)

	for sample_id, description in tqdm(descriptions.items(), total=len(descriptions), desc="Processing sample"):
		response = client.chat.completions.create(
			model="gpt-4o",
			messages=[
				{
					"role": "user",
					"content": get_vqa_extraction_prompt(description, mini=mini),
				}
			]
		)

		content = response.choices[0].message.content.strip()
		if  content.startswith("```"):
			content = content.strip("`")
			content = content.replace("json", "", 1).strip()

		json_response = json.loads(content)
		json_response["description"] = description
		with open(output_path/f"{sample_id}.json", "w") as fp:
			json.dump(json_response, fp, indent=2)

		add_entry_to_checkpoint(sample_id, checkpoint_path)


if __name__ == "__main__":
	parser = ArgumentParser(description="Extracts VQA and metadata for each questions given the description")

	parser.add_argument("--name", required=True)
	parser.add_argument("--input-json", required=True)
	parser.add_argument("--mini", action="store_true")

	args = parser.parse_args()
	
	# open the input file
	with open(args.input_json, "r") as fp:
		descriptions = json.load(fp)

	print(f"Total {len(descriptions)} data are loaded")
	
	base_path = Path("./outputs/qa-pairs")
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
		descriptions = {sample_id: description for sample_id, description in descriptions.items() if sample_id not in checkpoint_ids}
	else:
		with open(checkpoint_path, "w") as fp:
			json.dump([], fp)
	
	main(descriptions, output_path, checkpoint_path, mini=args.mini)
	