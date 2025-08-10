import json
import argparse
import os
import logging
import warnings
import torch
import tarfile
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from transformers import logging as transformers_logging
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# --- Modified Functions for Parallelism ---

def load_qwen_omni_model(model_path, device):
    """
    Loads the Qwen Omni model and processor onto a specific device.
    
    Args:
        model_path (str): Path to the pre-trained model.
        device (str): The device to load the model on (e.g., "cuda:0").
    """

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
    ).to(device)

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    return model, processor

def generate_answer_qwen_omni(model, processor, audio_path, disable_talker):
    """
    Generates an answer for a single instruction using the Qwen Omni model.
    """
    USE_AUDIO_IN_VIDEO = False
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path}
            ],
        }
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": 400,
        "do_sample": True,
        "temperature": 0.7,
    }
    if not disable_talker:
        text_ids, _ = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, speaker="Ethan", **gen_kwargs)
    else:
        text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, **gen_kwargs)

    response_text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response = response_text.split("\nassistant\n")[-1].strip()
    return response

# --- Worker Function with Individual Progress Bar ---

def process_chunk_on_gpu(chunk, device_id, model_path, data_dir, disable_talker):
    """
    This function is executed by each worker process on a specific GPU.
    It loads a model and processes a chunk of data, displaying its own progress bar.
    """
    device = f"cuda:{device_id}"
    processed_results = []
    
    try:
        model, processor = load_qwen_omni_model(model_path, device)
        if disable_talker:
            model.disable_talker()

        progress_bar = tqdm(
            chunk, 
            desc=f"GPU {device_id} (PID: {os.getpid()})", 
            position=device_id + 1, 
            leave=False,
        )
        
        for data in progress_bar:
            tar_dir = os.path.dirname(data["audio_path"]) 
            audio_path = os.path.basename(data["audio_path"]) 
            tar_path = os.path.join(data_dir, tar_dir, "data.tar")  


            temp_audio_path = os.path.join(data_dir, tar_dir, os.path.basename(audio_path))
            
            with tarfile.open(tar_path, "r") as tar:
                audio_file = tar.extractfile(audio_path)
                if audio_file is None:
                    raise FileNotFoundError(f"{audio_path} not found in {tar_path}")
                with open(temp_audio_path, "wb") as temp_file:
                    temp_file.write(audio_file.read()) 

            output = generate_answer_qwen_omni(model, processor, temp_audio_path, disable_talker)
            data["model_answer"] = output
            processed_results.append(data)
            
    except Exception as e:
        tqdm.write(f"Error in worker process {os.getpid()} on device {device}: {e}")
    
    return processed_results


def main(args):
    """
    Main function to orchestrate parallel processing.
    """
    mp.set_start_method('spawn', force=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found. This script requires at least one GPU.")
        return

    print(f"Found {num_gpus} GPUs. Starting parallel processing...")

    with open(args.input_meta_file, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    
    for i, item in enumerate(all_data):
        item['_original_index'] = i

    chunks = [[] for _ in range(num_gpus)]
    for i, data_item in enumerate(all_data):
        chunks[i % num_gpus].append(data_item)

    all_results = []
    disable_talker = True

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(process_chunk_on_gpu, chunk, i, args.model_path, args.data_dir, disable_talker): i 
                   for i, chunk in enumerate(chunks) if chunk} # Only submit non-empty chunks

        main_progress = tqdm(
            as_completed(futures), 
            total=len(futures), 
            desc="Overall Progress", 
            position=0,
        )

        for future in main_progress:
            try:
                result_chunk = future.result()
                all_results.extend(result_chunk)
            except Exception as e:
                gpu_id = futures[future]
                tqdm.write(f"A chunk processing on GPU {gpu_id} encountered a fatal error: {e}")

    all_results.sort(key=lambda x: x.get('_original_index', 0))
    
    for item in all_results:
        if '_original_index' in item:
            del item['_original_index']

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"\nAll {len(all_results)} model answers have been generated and saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen-Omni inference in parallel on multiple GPUs.")
    parser.add_argument("--input_meta_file", type=str, default="/mnt/data/panleyi/panleyi-data/Omni-SafetyBench-new/meta_files/unimodal/audio-only.jsonl", help="Input JSONL file")
    parser.add_argument("--output_file", type=str, default="/path/to/output_file", help="Output JSONL file")
    parser.add_argument("--data_dir", type=str, default="/mnt/data/panleyi/panleyi-data/Omni-SafetyBench-new/", help="Data directory (usage depends on full implementation)")
    parser.add_argument("--model_path", type=str, default="/mnt/data_cpfs/panleyi/Qwen2.5-Omni-7B", help="Path to the Qwen-Omni model")
    args = parser.parse_args()
    main(args)