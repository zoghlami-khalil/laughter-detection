import argparse
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def transcribe_audio(file_path, pipe, transcripts_dir):
    try:
        result = pipe(file_path, return_timestamps=True, generate_kwargs={"language": "english"})
        if result:
            txt_file_name = "audio.txt"  # Fixed output file name
            txt_file_path = os.path.join(transcripts_dir, txt_file_name)
            with open(txt_file_path, 'w') as txt_file:
                txt_file.write('\n'.join(str(chunk) for chunk in result["chunks"]))
            print(f"Transcription saved to {txt_file_path}")
            return txt_file_path
        else:
            print("No transcription was generated.")
            return None
    except ValueError as e:
        print(f"Error processing file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Audio Transcription with Whisper')
    parser.add_argument('path', type=str, help='Path to the audio file or directory.')
    args = parser.parse_args()

    transcripts_dir = "transcripts"
    create_directory(transcripts_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-base"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    if os.path.isdir(args.path):
        audio_files = [f for f in os.listdir(args.path) if f.endswith(('.mp3', '.wav', '.m4a'))]
        for filename in tqdm(audio_files, desc="Transcribing audio files"):
            file_path = os.path.join(args.path, filename)
            transcribe_audio(file_path, pipe, transcripts_dir)
    elif os.path.isfile(args.path):
        transcribe_audio(args.path, pipe, transcripts_dir)
    else:
        print("The provided path does not exist.")

if __name__ == "__main__":
    main()
