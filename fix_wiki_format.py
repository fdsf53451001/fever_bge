import csv

# Original CSV content is in 'input.csv' and the transformed content will be written to 'output.csv'
input_filename = 'result/testset_evidence_10_wikiapi.csv'
output_filename = 'result/testset_evidence_10_wikiapi_fixed.csv'

with open(input_filename, mode='r', newline='', encoding='utf-8') as infile, \
        open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:

    # Create CSV reader and writer
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Read the header and write the new header to the output CSV
    header = next(reader)
    new_header = ['id', 'claim'] + [f'evi{i}' for i in range(1, 11)]
    writer.writerow(new_header)

    # Iterate over the rows in the original CSV
    for row in reader:
        id, claim, evi1, evi2 = row
        # Parse the evidence list, ignore the brackets and split by ', ' to create a list
        evidence_list = evi2.strip("[]").split("', '")
        # Reformat evidence list by removing leading and trailing single quotes
        evidence_list = [evi.strip("'") for evi in evidence_list]
        # Write the new row to the output CSV
        writer.writerow([id, claim] + evidence_list)

# Output to the console that the transformation is complete
print(f'Transformation complete. Data is saved in {output_filename}.')