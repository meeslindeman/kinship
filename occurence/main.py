# Import necessary libraries for handling command line arguments
import argparse
from occ import load_data, analyze_co_occurrences, print_co_occurrences, count_symbols, count_mssg, analyze_all_symbols, invert_mapping, print_messages_per_category

def main():
    # Setup argparse to handle command line arguments
    parser = argparse.ArgumentParser(description="Process and analyze co-occurrence data.")
    parser.add_argument("--data_path", type=str, default='single_dataframe', help="Path to the data CSV file.")
    parser.add_argument("--print_occ", action='store_true', help="Print co-occurrence counts.")
    parser.add_argument("--print_mssg", action='store_true', help="Print messages per category.")
    parser.add_argument("--count_symbols", action='store_true', help="Count occurrences of each symbol.")
    parser.add_argument("--count_mssg", action='store_true', help="Count occurrences of each message.")
    parser.add_argument("--symbol_cat", action='store_true', help="Analyze all symbols for relationships, ages, and genders.")
    parser.add_argument("--cat_symbol", action='store_true', help="Analyze all symbols for relationships, ages, and genders.")
    args = parser.parse_args()

    # Load the data
    df = load_data(f'results/{args.data_path}.csv')
    combo_counts = analyze_co_occurrences(df)

    # Conditionally execute functions based on arguments
    if args.print_occ:
        print_co_occurrences(combo_counts)
        print("\n")
    
    if args.print_mssg:
        print_messages_per_category(combo_counts)
        print("\n")
    
    if args.count_symbols:
        count_symbols(combo_counts)

    if args.count_mssg:
        count_mssg(combo_counts)

    if args.symbol_cat:
        all_relationships, all_ages, all_genders = analyze_all_symbols(combo_counts)
        # Optionally print results
        for symbol in all_relationships:
            print(f"\nSymbol {symbol}:")
            print("Relationship Counts:")
            for relationship, count in all_relationships[symbol].items():
                print(f"{relationship}: {count} times")
            print("Age Counts:")
            for age, count in all_ages[symbol].items():
                print(f"{age}: {count} times")
            print("Gender Counts:")
            for gender, count in all_genders[symbol].items():
                print(f"{gender}: {count} times")
        print("\n")

    if args.cat_symbol:
        all_relationships, all_ages, all_genders = analyze_all_symbols(combo_counts)

        # Usage of the dictionaries to create inverted mappings
        inverted_relationships = invert_mapping(all_relationships)
        inverted_ages = invert_mapping(all_ages)
        inverted_genders = invert_mapping(all_genders)

        # Printing results
        print("\nRelationships by Symbol:")
        for relationship, symbols in inverted_relationships.items():
            print(f"\n{relationship}:")
            for symbol, count in sorted(symbols.items(), key=lambda item: item[1], reverse=True):
                print(f"Symbol {symbol}: {count} times")

        print("\nAges by Symbol:")
        for age, symbols in inverted_ages.items():
            print(f"\n{age}:")
            for symbol, count in sorted(symbols.items(), key=lambda item: item[1], reverse=True):
                print(f"Symbol {symbol}: {count} times")

        print("\nGenders by Symbol:")
        for gender, symbols in inverted_genders.items():
            print(f"\n{gender}:")
            for symbol, count in sorted(symbols.items(), key=lambda item: item[1], reverse=True):
                print(f"Symbol {symbol}: {count} times")

if __name__ == "__main__":
    main()
