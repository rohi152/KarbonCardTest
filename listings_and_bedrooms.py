
import json
import numpy as np
import pandas as pd

SAMPLE_FILE_PATH = "listing_data/sample_listing.json"

def load_data_from_json(file_path):

    with open(file_path,'r') as f:
        data = json.load(f)

    return data


def evaluate_listing(desc_text, actual_count):
    x = desc_text.lower()
    c = int(actual_count)

    if ('art studio' in x) or ('yoga studio' in x):
        if "studio with" in x:
            return 0
        else:
            return c
    elif ('studio' not in x) and ('1-bedroom' not in x):
        return c
    else:
        return c

class __main__():
    listing_json = load_data_from_json(SAMPLE_FILE_PATH)
    listing_data = pd.DataFrame(listing_json)
    listing_data['corrected_count'] = listing_data.apply(lambda row: evaluate_listing(row['description'],row['num_bedrooms']), axis=1)
    print(listing_data['corrected_count'])