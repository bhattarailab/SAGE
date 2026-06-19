import argparse
import easyocr
import numpy as np
import os
import pandas as pd
import re

TIME_REGEX = r"(?:[01]\d|2[0-3])[.:][0-5]\d[.:][0-5]\d" 
DATE_REGEX = r"(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/\d{4}"

# OCR output = list(tuples with 3 element) where, len(output) = number of text
# each text tuple will have = ([topleft, topright, bottomright, bottomleft], text, confidence)
# the coordinate for each corner will be [x, y] where, top-leftmost part is considered (0, 0)

def top_y(r):
    return r[0][0][1]

def left_x(r):
    return r[0][0][0]

def right_x(r):
    return r[0][1][0]

def height(r):
    return r[0][3][1] - r[0][0][1]



def main(csv_path, output_path):
    df = pd.read_csv(csv_path)
    # default timestamp for all the samples
    df["timestamp"] = None

    # intializing the reader
    reader = easyocr.Reader(["en"], gpu=False)
    
    for i, df_row in df.iterrows():
        results = reader.readtext(df_row.file_path)
              
        # sort the results by the vertical positioning with tolerance
        sorted_results = sorted(results, key=top_y)
        if len(sorted_results) == 0:
            continue

        # computing the vertical tolerance
        vertical_tolerance = np.median([r[0][3][1]-r[0][0][1] for r in results])*0.5

        rows = []
        current_row = [sorted_results[0]]

        # vertical separation
        for r in sorted_results[1:]:
            if abs(top_y(r) - top_y(current_row[0])) <= vertical_tolerance:
                current_row.append(list(r))
            else:
                rows.append(current_row)
                current_row = [r]
        
        rows.append(current_row)

        # horizontal sorting
        for row in rows:
            row.sort(key=left_x)

        # horizontal separation
        horizontal_threshold = np.median([(right_x(r)-left_x(r))/len(r[1]) for r in results if len(r[1]) > 0])
                    
        # sort the results by the horizontal positioning while creaing groups with threshold
        canvas_lines = []

        for row in rows:
            merged = []
            for r in row:
                if not merged:
                    merged.append(list(r))
                    continue
                
                prev = merged[-1]
                gap = left_x(r)-right_x(prev)

                if gap < horizontal_threshold:
                    prev[1] = prev[1]+r[1]
                    prev[2] *= r[2]
                    prev[0] = (
                        (left_x(prev), top_y(prev)),
                        (right_x(r), top_y(r)),
                        (right_x(r), top_y(r)+height(r)),
                        (left_x(prev), top_y(prev)+height(prev)),
                    )
                else:
                    merged.append(r)

            canvas_lines.append(merged)

        text = "\n".join([" ".join([r[1] for r in line]) for line in canvas_lines])
        time_match  = re.search(TIME_REGEX, text)
        date_match = re.search(DATE_REGEX, text)

        time = time_match.group() if time_match else None
        date = date_match.group() if date_match else None

        if time is not None and date is not None:
            date = list(date)
            date[2] = "/"
            date[5] = "/"
            date = "".join(date)
            
            time = list(time)
            time[2] = "."
            time[5] = "."
            time = "".join(time)

            df.loc[i, "timestamp"] = f"{date} {time}"
    
    print(f"You have to edit {sum(df['timestamp'].isna())} samples")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="path to CSV created from create-non-duplicate-entries.py")
    parser.add_argument("--output", "-o", help="output path for the output csv")

    args = parser.parse_args()
    csv_path, output_path = args.csv, args.output

    if not os.path.exists(csv_path):
        raise Exception(f"CSV path not found, got `{csv_path}`")

    main(csv_path=csv_path, output_path=output_path)
