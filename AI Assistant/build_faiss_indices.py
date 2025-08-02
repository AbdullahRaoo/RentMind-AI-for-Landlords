import os
import sys
sys.path.append(os.path.dirname(__file__))
from faiss_utils import build_faiss_index

# Paths for cleaned and raw data
CLEANED_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent_Pricing_AI/data/cleaned_rent_data.csv'))
CLEANED_INDEX = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent_Pricing_AI/data/cleaned_rent_data.faiss'))
RAW_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent_Pricing_AI/data/rent_ads_rightmove_extended.csv'))
RAW_INDEX = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Rent_Pricing_AI/data/rent_ads_rightmove_extended.faiss'))

def main():
    print('Building FAISS index for cleaned data...')
    build_faiss_index(CLEANED_CSV, CLEANED_INDEX)
    print('Done with cleaned data.')
    if os.path.exists(RAW_CSV):
        print('Building FAISS index for raw data...')
        build_faiss_index(RAW_CSV, RAW_INDEX)
        print('Done with raw data.')
    else:
        print('Raw data CSV not found, skipping.')

if __name__ == "__main__":
    main()
