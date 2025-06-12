import argparse

def filter_bed_by_length(input_bed, output_bed, target_length=100):
    """
    Filters BED entries to keep only those with the specified length (default: 100bp).
    """
    with open(input_bed, 'r') as infile, open(output_bed, 'w') as outfile:
        for line in infile:
            if line.strip() == "" or line.startswith("#"):
                continue  # skip empty lines and comments
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue  # malformed line

            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue  # skip lines with invalid start/end

            if (end - start) == target_length:
                outfile.write(line)

    print(f"Saved filtered BED with length={target_length} to: {output_bed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter BED file for sequences of specific length.")
    parser.add_argument("input_bed", help="Path to input BED file")
    parser.add_argument("output_bed", help="Path to output BED file")
    parser.add_argument("--length", type=int, default=100, help="Target sequence length (default: 100)")

    args = parser.parse_args()
    filter_bed_by_length(args.input_bed, args.output_bed, args.length)
