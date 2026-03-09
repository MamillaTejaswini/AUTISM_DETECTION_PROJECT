import arff
import pandas as pd

input_arff = "../Data/Autism-Child-Data.arff"
output_csv = "../Data/Autism-Child-Data.csv"

with open(input_arff, 'r') as f:
    arff_data = arff.load(f)

df = pd.DataFrame(
    arff_data['data'],
    columns=[attr[0] for attr in arff_data['attributes']]
)

df.to_csv(output_csv, index=False)

print("ARFF file successfully converted to CSV!")

