import gradio as gr
import subprocess
import os
import shutil
# Set the environment variable for the temporary directory to a path you control
os.environ['TMPDIR'] = 'laughter-detection/tmp'

# Ensure the directory exists
os.makedirs(os.environ['TMPDIR'], exist_ok=True)

def cleanup_previous_data():
    """Delete files and directories from previous runs."""
    paths_to_clean = [
        'chunks',
        'transcripts',
        'word_transcripts',
        'laughter_timestamps.txt',
        'timestamps.txt',
        'updated_transcript.txt'
    ]
    for path in paths_to_clean:
        if os.path.isdir(path):
            shutil.rmtree(path)  # Remove directory and all its contents
        elif os.path.isfile(path):
            os.remove(path)  # Remove a file

def format_timestamp(start, end):
    """Converts timestamps from seconds to 'minute:second' format."""
    start_min, start_sec = divmod(int(start), 60)
    end_min, end_sec = divmod(int(end), 60)
    return f"[{start_min:02}:{start_sec:02}-{end_min:02}:{end_sec:02}]"

def process_audio(audio_file_path):
    cleanup_previous_data()
    if not audio_file_path:
        return "No file received. Please upload a valid audio file."
    
    # Run the transcribe.py script
    subprocess.run(["python3", "laughter-detection/transcribe.py", audio_file_path, "2000"], check=True)
    # Run the transcribe_sentence.py script
    subprocess.run(["python3", "laughter-detection/transcribe_sentence.py", audio_file_path], check=True)
    
    try:
        subprocess.run([
            "python3",
            "laughter-detection/timestamp_laughter.py",  # Ensure the path is correct
            "--input_audio_dir", "chunks",
            "--timestamps_file", "timestamps.txt"
        ], check=True, capture_output=False)  # Added capture_output=False to see outputs directly
    except subprocess.CalledProcessError as e:
        print("Error running script:", e)
        print("Return code:", e.returncode)
        print("Output:", e.output.decode())

    # Run the update_transcript.py script
    subprocess.run(["python3", "laughter-detection/update_transcript.py"], check=True)

    formatted_transcript = ""
    transcript_path = "updated_transcript.txt"
    with open(transcript_path, "r") as file:
        for line in file:
            data = eval(line.strip())
            formatted_timestamp = format_timestamp(*data['timestamp'])
            formatted_transcript += f"{formatted_timestamp} {data['text'].strip()}\n"

    return formatted_transcript.strip(), transcript_path

# Set up the Gradio interface
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs=[gr.Textbox(label="Transcript"), gr.File(label="Download Transcript")],
    title="Enhanced Audio Transcripts Part1: Laughter Detection",
    description="Upload an audio file of your choice (any length) and click 'Submit' to process. You can also record your own voice.",
    allow_flagging="never",
)

iface.launch()