import argparse
import os
import ast
import torch
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    logging.debug(f"Directory created at: {path}")

def transcribe_audio(file_path, pipe, transcripts_dir):
    try:
        result = pipe(file_path, return_timestamps="word", generate_kwargs={"language": "english"})
        if result:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            txt_file_name = base_name + ".txt"
            txt_file_path = os.path.join(transcripts_dir, txt_file_name)
            with open(txt_file_path, 'w') as txt_file:
                txt_file.write('\n'.join(str(chunk) for chunk in result["chunks"]))
            logging.info(f"Transcription saved to {txt_file_path}")
            return txt_file_path
        else:
            logging.warning("No transcription was generated.")
            return None
    except ValueError as e:
        logging.error(f"Error processing file {file_path} : {e}")
        return None

def check_timestamps_and_slice(audio_file_path, transcript_file_path, chunks_dir, jump_threshold, timestamp_records):
    with open(transcript_file_path, 'r') as file:
        lines = file.readlines()
        rows = [ast.literal_eval(line.strip()) for line in lines if line.strip()]

    timestamps = []
    for i in range(len(rows) - 1):
        current_end_time = rows[i]['timestamp'][1] * 1000 if rows[i]['timestamp'][1] is not None else None
        next_start_time = rows[i + 1]['timestamp'][0] * 1000 if rows[i+1]['timestamp'][1] is not None else None
        if current_end_time is not None and next_start_time is not None and (next_start_time - current_end_time > jump_threshold):
            timestamps.append((int(current_end_time), int(next_start_time)))
            logging.debug(f"Timestamp slice between {current_end_time} and {next_start_time} identified.")
        else:
            logging.debug(f"Skipping segment due to missing or insufficient gap in timestamp: current_end_time={current_end_time}, next_start_time={next_start_time}")

    episode_name = os.path.splitext(os.path.basename(transcript_file_path))[0]
    episode_dir = chunks_dir
    create_directory(episode_dir)

    audio = AudioSegment.from_file(audio_file_path)
    for i, (start_ms, end_ms) in enumerate(timestamps):
        sliced_audio = audio[start_ms:end_ms]
        output_file_name = f'sliced_chunk_{i+1}.wav'
        output_file_path = os.path.join(episode_dir, output_file_name)
        sliced_audio.export(output_file_path, format="wav")
        timestamp_records.append(f"{output_file_name} {start_ms / 1000.0}")
        logging.info(f"Sliced audio chunk saved as {output_file_name}")

    logging.info(f"Slicing complete for {episode_name}.")

def main():
    parser = argparse.ArgumentParser(description='Audio Processing with Whisper and PyDub')
    parser.add_argument('path', type=str, help='Path to the audio file or directory.')
    parser.add_argument('jump_threshold', type=int, help='Threshold for timestamp jumps in milliseconds.')
    args = parser.parse_args()

    transcripts_dir = "word_transcripts"
    chunks_dir = "chunks"
    timestamp_records = []
    create_directory(transcripts_dir)
    create_directory(chunks_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-base"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, max_new_tokens=128, chunk_length_s=30, batch_size=16, return_timestamps=True, torch_dtype=torch_dtype, device=device)

    if os.path.isdir(args.path):
        audio_files = [f for f in os.listdir(args.path) if f.endswith(('.mp3', '.wav', '.m4a'))]
        logging.info(f"Processing {len(audio_files)} audio files in directory {args.path}")
        for filename in tqdm(audio_files, desc="Processing audio files"):
            file_path = os.path.join(args.path, filename)
            transcript_file_path = transcribe_audio(file_path, pipe, transcripts_dir)
            if transcript_file_path:
                check_timestamps_and_slice(file_path, transcript_file_path, chunks_dir, args.jump_threshold, timestamp_records)
    elif os.path.isfile(args.path):
        logging.info(f"Processing single audio file at {args.path}")
        transcript_file_path = transcribe_audio(args.path, pipe, transcripts_dir)
        if transcript_file_path:
            check_timestamps_and_slice(args.path, transcript_file_path, chunks_dir, args.jump_threshold, timestamp_records)
    else:
        logging.error("The provided path does not exist.")

    with open("timestamps.txt", 'w') as f:
        for record in timestamp_records:
            f.write(record + '\n')
    logging.info("Timestamps recorded in 'timestamps.txt'.")

if __name__ == "__main__":
    main()
