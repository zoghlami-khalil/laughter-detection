import os, sys, pickle, time, librosa, argparse, torch, numpy as np, pandas as pd, scipy
from tqdm import tqdm
import tgt
import logging  # Import logging library

# Setup basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
utils_path = os.path.join(script_dir, 'utils')
sys.path.append(utils_path)

import laugh_segmenter
import models, configs
import dataset_utils, audio_utils, data_loaders, torch_utils
from torch import optim, nn
from functools import partial
from distutils.util import strtobool
import glob

model_path = os.path.join(script_dir, 'checkpoints', 'in_use', 'resnet_with_augmentation')

sample_rate = 8000

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=model_path)
parser.add_argument('--config', type=str, default='resnet_with_augmentation')
parser.add_argument('--threshold', type=str, default='0.5')
parser.add_argument('--min_length', type=str, default='0.2')
parser.add_argument('--input_audio_dir', required=True, type=str)
parser.add_argument('--timestamps_file', required=True, type=str)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--save_to_audio_files', type=str, default='False')
parser.add_argument('--save_to_textgrid', type=str, default='False')

args = parser.parse_args()

# Log all arguments used
logging.info(f"Arguments used: {args}")

model_path = args.model_path
config = configs.CONFIG_MAP[args.config]
audio_dir = args.input_audio_dir
threshold = float(args.threshold)
min_length = float(args.min_length)
save_to_audio_files = bool(strtobool(args.save_to_audio_files))
save_to_textgrid = bool(strtobool(args.save_to_textgrid))
output_dir = args.output_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

##### Load the Model

model = config['model'](dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
feature_fn = config['feature_fn']
model.set_device(device)

if os.path.exists(model_path):
    torch_utils.load_checkpoint(model_path+'/best.pth.tar', model)
    model.eval()
    logging.info("Model loaded and set to evaluation mode.")
else:
    logging.error(f"Model checkpoint not found at {model_path}")
    raise Exception(f"Model checkpoint not found at {model_path}")

##### Read Initial Timestamps File

initial_timestamps = {}
with open(args.timestamps_file, 'r') as file:
    for line in file:
        parts = line.split()
        initial_timestamps[parts[0]] = float(parts[1])

##### Process each file in the directory

all_laughter_timestamps = []
audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
logging.info(f"Found {len(audio_files)} audio files in directory {audio_dir}.")

# Before iterating over the files, log the contents of the directory
if os.path.exists(audio_dir):
    logging.info(f"Directory '{audio_dir}' exists. Contents: {os.listdir(audio_dir)}")
else:
    logging.error(f"Directory '{audio_dir}' does not exist.")


for audio_file in audio_files:
    file_name = os.path.basename(audio_file)
    logging.info(f"Processing file: {audio_file}")

    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=audio_file, feature_fn=feature_fn, sr=sample_rate)

    collate_fn = partial(audio_utils.pad_sequences_with_labels, expand_channel_dim=config['expand_channel_dim'])

    inference_generator = torch.utils.data.DataLoader(
        inference_dataset, num_workers=4, batch_size=8, shuffle=False, collate_fn=collate_fn)

    ##### Make Predictions

    probs = []
    for model_inputs, _ in tqdm(inference_generator):
        x = torch.from_numpy(model_inputs).float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        if len(preds.shape) == 0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    probs = np.array(probs)

    file_length = audio_utils.get_audio_length(audio_file)
    fps = len(probs) / float(file_length)
    probs = laugh_segmenter.lowpass(probs)
    instances = laugh_segmenter.get_laughter_instances(probs, threshold=threshold, min_length=min_length, fps=fps)

    base_time = initial_timestamps.get(file_name, 0)

    if len(instances) > 0:
        for instance in instances:
            adjusted_start = base_time + instance[0]
            adjusted_end = base_time + instance[1]
            all_laughter_timestamps.append((file_name, adjusted_start, adjusted_end))
            logging.debug(f"Detected laughter: {file_name} from {adjusted_start} to {adjusted_end}")

# Writing all laughter timestamps to a file
with open('laughter_timestamps.txt', 'w') as outfile:
    outfile.write("All laughter timestamps:\n")
    for filename, start, end in all_laughter_timestamps:
        outfile.write(f"File {filename}: Laughter from {start}s to {end}s\n")
    logging.info("All laughter timestamps have been written to laughter_timestamps.txt")