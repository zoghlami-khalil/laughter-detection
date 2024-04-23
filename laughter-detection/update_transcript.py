import ast

def read_transcripts(filename):
    with open(filename, 'r') as file:
        transcripts = [ast.literal_eval(line.strip()) for line in file if line.strip()]
    return transcripts

def read_laughter_events(filename):
    laughter_events = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                # Extract time information from the line after splitting on ':'
                parts = line.split(':')
                if len(parts) > 1:
                    # Extract the numerical start time directly after 'Laughter from'
                    time_info = parts[1].strip()  # This gets the part after the colon
                    start_time_str = time_info.split(' to ')[0]  # Splits on 'to' and takes the first part
                    start_time_str = start_time_str.replace('s', '')  # Removes 's'
                    start_time_str = start_time_str.replace('Laughter from ', '')  # Removes 'Laughter from '
                    start_time = float(start_time_str)
                    laughter_events.append(start_time)
            except ValueError as e:
                print(f"Error converting to float: {start_time_str}, Error: {e}")
    return laughter_events

def integrate_laughter(transcripts, laughter_events):
    for event_start in laughter_events:
        for i, entry in enumerate(transcripts):
            start, end = entry['timestamp']
            if start <= event_start <= end:
                entry['text'] += " (Laughing)"
                break
            elif i < len(transcripts) - 1 and event_start > end and event_start < transcripts[i + 1]['timestamp'][0]:
                entry['text'] += " (Laughing)"
                break

def write_updated_transcripts(transcripts, filename):
    with open(filename, 'w') as file:
        for entry in transcripts:
            file.write(f"{entry}\n")

def main():
    # You can change these paths to relative paths or pass them as arguments
    transcripts_file = 'transcripts/audio.txt'
    laughter_file = 'laughter_timestamps.txt'
    output_file = 'updated_transcript.txt'  # Output directly to the local directory

    transcripts = read_transcripts(transcripts_file)
    laughter_events = read_laughter_events(laughter_file)
    integrate_laughter(transcripts, laughter_events)
    write_updated_transcripts(transcripts, output_file)
    print(f"Updated transcripts written to {output_file}")

if __name__ == "__main__":
    main()