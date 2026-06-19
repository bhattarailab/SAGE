import json
import numpy as np

models_and_their_result_files = {
  "gemini-3-flash": "./outputs/benchmarking/results_gemini-3-flash-guthalu.json",
  "grok-4.3": "./outputs/benchmarking/results_grok-4.3-guthalu.json",
  "gemma-4-31b": "./outputs/benchmarking/results_gemma-4-31b-guthalu.json",
  "claude-sonnet-4.6": "./outputs/benchmarking/claude-sonnet-4.6-guthalu.json",
  "qwen-2.5vl-72b": "./outputs/benchmarking/qwen2.5vl-72b-guthalu.json",
}

models = list(models_and_their_result_files.keys())
categories = ["anatomical landmark", "polyp", "instrument", "abnormality", "inflammation", "infection"]
cat_labels = categories

model_stats = dict()

def compute_average_green_score(file_path):
  with open(file_path, "r") as fp:
    data = json.load(fp)

  # mean green score
  mean_green = (np.array([d["green"] for d in data])).mean()
  
  # for each questions
  question_green = {}
  for q_key, q_type in zip(["q1", "q2", "q3", "q4", "q5", "q6"], categories):
    green = np.array([d["green"] for d in data if d["question_key"] == q_key])
    question_green[q_type] = (green).mean()
  
  return {
    "mean": mean_green,
    **question_green,
  }

# compute all the stats
for model, result_path in models_and_their_result_files.items():
  green_score = compute_average_green_score(result_path)

  model_stats[model] = green_score

print(model_stats)