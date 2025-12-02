import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer


def xes_to_csv_with_traces(xes_file_path, csv_output_path):
    """
    Convert XES event log to CSV with CaseID, trace, and length columns.

    Parameters:
    xes_file_path (str): Path to the input XES file
    csv_output_path (str): Path to the output CSV file
    """

    # Import the XES file
    log = xes_importer.apply(xes_file_path)

    # List to store data for each case
    cases_data = []

    # Process each trace in the log
    for trace in log:
        # Get case ID from trace attributes
        case_id = trace.attributes['concept:name']

        # Extract event names from the trace
        events = []
        for event in trace:
            event_name = event['concept:name']
            events.append(event_name)

        # Calculate length (number of actual events, excluding BOS and EOS)
        trace_length = len(events)

        # Add BOS and EOS markers
        trace_string = '<BOS>, ' + ', '.join(events) + ', <EOS>'

        # Store in list as dictionary
        cases_data.append({
            'CaseID': case_id,
            'trace': trace_string,
            'length': trace_length
        })

    # Create DataFrame
    df = pd.DataFrame(cases_data)

    # Save to CSV
    df.to_csv(csv_output_path, index=False)

    print(f"CSV file saved to: {csv_output_path}")
    print(f"Total cases processed: {len(df)}")
    print(f"Average trace length: {df['length'].mean():.2f}")
    print(f"Min trace length: {df['length'].min()}")
    print(f"Max trace length: {df['length'].max()}")

    return df


# Example usage
if __name__ == "__main__":
    # Specify your file paths
    input_xes_file = "data/helpdesk_20.xes"  # Replace with your XES file path
    output_csv_file = "data/helpdesk_20.csv"  # Replace with desired output path

    # Process the XES file
    df_result = xes_to_csv_with_traces(input_xes_file, output_csv_file)

    # Display first few rows to verify
    print("\nFirst 5 rows of the output:")
    print(df_result.head())

    # Optional: Display a sample trace
    if len(df_result) > 0:
        print(f"\nSample trace for case '{df_result.iloc[0]['CaseID']}':")
        print(df_result.iloc[0]['trace'])