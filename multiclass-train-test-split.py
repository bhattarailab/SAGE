"""
Script to get the expert mulit-class label annotation and agrregate it to the standard classes. Finally, utilize this class to create train and test splits of patients
"""

import json
import pandas as pd
from itertools import chain
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

SEED = 0
TEST_SIZE = 0.30

# improt the annotation given by GI expoerts
with open("./outputs/descriptions/final_annotations.json", "r") as fp:
  data = json.load(fp)

# extract sample_id: class
parsed_classes = {sample_id: value["classes"] for sample_id, value in data.items()}

MAPPINGS = {
  # anatomical landmark
  "appendix-orfice": ["appendix-orfice"],
  "cardia": ["cardia"],
  "cecum/ileum": ["cecum", "ileum"],
  "corpus": ["corpus"],
  "fundus": ["fundus"],
  "gej": ["gej"],
  "incisura": ["incisura"],
  "pylorus": ["antrum", "pylorus"],
  "rectum": ["anal-canal", "rectum", "rectal-lumen"],
  "z-line": ["z-line"],

  # section
  "duodenum": ["duodenum-mucosa", "duodenum-lumen", "duodenum-fold", "duodenum", "duodenual-fold", "duodenal-papilla", "duodenal-lumen", "duodenal-fold", "duodeanl-fold", "dudoenum", "dudoenal-lumen", "dudodenal-lumen", "d1", "d2"],
  "pharynx": ["pharynx", "oropharynx", "hypopharynx"],
  "esophagus": ["lower-esophagus", "esophagus", "esophageal-wall", "esophageal-mucosa", "esophageal-lumen"],
  "colon": ["transverse-colon", "descending-colon", "colon", "colonic-fold", "colonic-lumen", "colonic-mucosa", "colonic-wall", "ascending-colon"],
  "stomach": ["stomach", "stomach-mucosa", "stomach-wall", "rugal-fold", "gastric-mucosa", "gastric-fold", "gastric-body"],
  "small-intestine": ["small-intestine", "villi"],
  "ileocecal-valve": ["ileocecal-valve"],

  # abnormality
  "candidiasis": ["candidiasis"],
  "blood": ["bleeding", "blood", "active-bleeding"],
  "diverticulum": ["diverticulum"],
  "erosion": ["erosion"],
  "esophagitis": ["esophagitis"],
  "erythema": ["erythema", "diffuse-erythema"],
  "gave": ["gave"],
  "growth": ["ulcero-proliferative-growth", "mucosal-growth", "mucosal-bulge", "mass", "lesion", "bulge", "growth"],
  "hemorroid": ["hemorroid"],
  "inflammation": ["inflammation", "erythema"],
  "polyp": ["polyp", "polypoidal-lesion", "polypectomy", "pedunculated-polyp", "normal-polyp", "numerous-polyp", "multiple-polyp", "lobulated-polyp", "clustered-polyp", "resected-polyp", "polypoid-structure"],
  "ulcer": ["ulcero-proliferative-growth", "ulcerative-colitis", "ulcer", "healed-ulcer"],
  "varices": ["varices"],
  "worms": ["worms"],
  "others": ["coagulated-tissue", "linear-erythematous-lesion", "structural-abnormality", "haitus-hernia", "white-spots"],

  # instrument
  "apc-probe": ["apc-probe"],
  "biopsy-forceps": ["biopsy-forceps"],
  "cbd-stent": ["nasogastric-tube", "cbd-stent"],
  "endoscope": ["scope", "endoscope"],
  "hemoclip": ["hemoclip"],
  "ligation-band": ["ligation-band"],
  "peg-tube": ["peg-tube"],
  "rat-forceps": ["rat-forceps"],
  "instrument": ["instrument", "cre-balloon", "needle", "snare", "apc-probe", "biopsy-forceps", "scope", "endoscope", "hemoclip", "ligation-band", "peg-tube", "nasogastric-tube", "cbd-stent", "rat-forceps"],
  
  "foreign-body": ["foreign-body"],
  "food-residue": ["food-residue", "food-particle"],
  "fecal-matter": ["fecal-matter"],

  "outside-body": ["outside-body"],
  "no-view": ["no-view"],
  "good-view": ["good-view"],
  "moderate-view": ["moderate-view", "moderate-visibility"],
  "nbi": ["nbi"],
  "near-focus": ["near-focus"],

  # just for information
  "d1": ["d1"],
  "d2": ["d2"],
  "transverse-colon": ["transverse-colon"],
  "ascending-colon": ["ascending-colon"],
  "descending-colon": ["descending-colon"],
  "resected-polyp": ["resected-polyp", "polypectomy"],
  "sessile-polyp": ["sessile-polyp"],
  "pedunculated-polyp": ["pedunculated-polyp"],
  "numerous-polyp": ["numerous-polyp", "multiple-polyp"],
  "lobulated-polyp": ["lobulated-polyp"],
}
ALL_BOARDER_CLASSES = list(chain.from_iterable(MAPPINGS.values()))
# these are the classes that we find either highly ambiguous, or having very less number of samples < 3 or with no medical significance
SKIP_BOARDER_CLASSES = ["normal", "bile-stain", "bubbles", "oral-cavity", "remove", "bookmark", "boomark", "mucosa", "prominent-blood-vessels", "infection", "haustral-fold", "plaques", "mucosal-irregularity", "swelling", "irregular-mucosa", "longitudnal-fold", "ridgs", "ileal-lining", "ilela-mucosa", "splenic-fixture"]

def get_class_presence(classes, mapping):
  class_presence_dict = dict()
  for class_name, borader_name in mapping.items():
    for c in classes:
      if c not in ALL_BOARDER_CLASSES and c not in SKIP_BOARDER_CLASSES:
        raise Exception(f"Unknown broader name from the annotator, {c}")
    
    class_presence_dict[class_name] = any([c in borader_name for c in classes])

  return class_presence_dict


sample_class_presence_list = list()
for sample_id, classes in parsed_classes.items():
  sample_class_presence_list.append({
    "sample_id": sample_id,
    **get_class_presence(classes, MAPPINGS),
  })

df = pd.DataFrame(sample_class_presence_list)
classes_df = df.drop(columns="sample_id")
classes_counts = classes_df.sum().sort_values(ascending=False)

LABELS = ["stomach", "esophagus", "colon", "duodenum", "pylorus", "gej", "fundus", "z-line", "rectum", "polyp", "diverticulum", "inflammation", "growth", "esophagitis", "ulcer", "blood", "foreign-body", "instrument"]

df_labels = pd.DataFrame({
  "sample_uuid": df["sample_id"],
  "labels": df[LABELS].apply(lambda row: [label for label in LABELS if row[label]], axis=1)
})
df_labels.to_csv("./outputs/sample_labels.csv", index=False)

# classes_counts for sections
# stomach               372 x
# esophagus             232 x
# colon                 161 x
# duodenum               53 x

# for the anatomical landmark
# pylorus               258 x
# gej                    80 x
# fundus                 49 x
# z-line                 51 x
# rectum                 30 x

# for the view
# good-view             843 x
# moderate-view         163 x
# no-view                76 x
# nbi                    71 x
# outside-body           21 x

# for the abnormality
# polyp                 164 x
# diverticulum           23 x
# erythema               64 x 
# inflammation          142 x
# esophagitis            40 x
# ulcer                  39 x
# blood                  40 x
# foreign-body           38 x
# growth                 37 x
# worms                  17 x

# for the polyps type
# sessile-polyp         108 x

# for the instruments
# other-instrument       26 x
# endoscope             154 x

# for reporting in diagrams only
# cardia                 20 x
# biopsy-forceps         18 x
# cecum/ileum            17 x
# incisura               17 x
# food-residue           15 x
# ligation-band          13
# pedunculated-polyp     13 x
# varices                13 x
# resected-polyp         11 x
# apc-probe              11 x
# d2                     11 x
# near-focus             11
# hemorroid               9 x
# others                  9 x
# cbd-stent               8 x
# erosion                 8
# appendix-orfice         7 x
# d1                      7 x
# rat-forceps             7 x
# fecal-matter            7
# peg-tube                6 x
# ileocecal-valve         6
# small-intestine         6 x
# pharynx                 5 x
# hemoclip                4 x
# gave                    3
# candidiasis             3 x
# corpus                  3
# transverse-colon        3
# numerous-polyp          3 x
# ascending-colon         3 
# lobulated-polyp         3 x
# descending-colon        2

# fetch the sample_id and patient_id information
dataset_df = pd.read_csv("./outputs/complete-metadata.csv")
valid_df = dataset_df[dataset_df["sample_uuid"].isin(df["sample_id"])]
valid_patient_df = valid_df[["sample_uuid", "patient_uuid"]]

# merge the patient information and the class information
mcc_with_patients_df = df.merge(valid_patient_df, left_on="sample_id", right_on="sample_uuid", how="left")

# group the samples by patient and know the count of each class
patient_class_count_df = mcc_with_patients_df.groupby("patient_uuid")[list(MAPPINGS.keys())].sum().reset_index()

# perform the multilabel stratitified sampling
X = patient_class_count_df["patient_uuid"].values
y = (patient_class_count_df[list(MAPPINGS.keys())] > 0).astype(int).values

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_idx, test_idx = next(msss.split(X, y))

train_df = patient_class_count_df.iloc[train_idx].reset_index(drop=True)
test_df = patient_class_count_df.iloc[test_idx].reset_index(drop=True)
print(f"Number of patients in train and test df are {len(train_df)} and {len(test_df)}")

# validation by checking the proportion of each class
report = pd.DataFrame({
  "train_count": train_df[list(MAPPINGS.keys())].sum(),
  "test_count": test_df[list(MAPPINGS.keys())].sum(),
})

total = patient_class_count_df[list(MAPPINGS.keys())].sum()
report["train_%"] = (report["train_count"] / total * 100).round(1)
report["test_%"] = (report["test_count"] / total * 100).round(1)

report = report.reset_index().rename(columns={"index": "class"})
report.to_csv("./outputs/train-test-split-report.csv", index=False)

# add the train or test split logic into the final dataset csv
valid_df["split"] = valid_df["patient_uuid"].map(lambda x: "train" if x in set(train_df["patient_uuid"]) else "test")
valid_df.to_csv("./outputs/complete-dataset.csv", index=False)
