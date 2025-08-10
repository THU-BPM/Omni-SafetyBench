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
import shutil
from transformers import logging as transformers_logging
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


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

def generate_answer_qwen_omni(model, processor, video_file, disable_talker):
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
                {"type": "video", "video": video_file}
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
    Modified worker function that properly handles OOM errors.
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
            try:
                tar_dir = os.path.dirname(data["video_path"])
                video_path = os.path.basename(data["video_path"])
                tar_path = os.path.join(data_dir, tar_dir, "data.tar")
                temp_video_path = os.path.join(data_dir, tar_dir, os.path.basename(video_path))

                with tarfile.open(tar_path, "r") as tar:
                    video_file = tar.extractfile(video_path)
                    if video_file is None:
                        raise FileNotFoundError(f"{video_path} not found in {tar_path}")
                    with open(temp_video_path, "wb") as temp_file:
                        shutil.copyfileobj(video_file, temp_file)

                output = generate_answer_qwen_omni(model, processor, temp_video_path, disable_talker)

                data["model_answer"] = output
                processed_results.append(data)
                
            except Exception as e:
                tqdm.write(f"Error processing item on GPU {device_id}: {e}")
                # Add the failed item to results with error message
                data["model_answer"] = f"ERROR: {str(e)}"
                processed_results.append(data)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        tqdm.write(f"GPU {device_id} encountered OOM error. Marking device as failed.")
        return {"status": "OOM", "device_id": device_id, "processed_results": processed_results}
        
    except Exception as e:
        tqdm.write(f"Fatal error in worker process {os.getpid()} on device {device}: {e}")
        return {"status": "ERROR", "device_id": device_id, "processed_results": processed_results}
    
    return {"status": "SUCCESS", "device_id": device_id, "processed_results": processed_results}

def main(args):
    """
    Modified main function that handles GPU failures and chunk redistribution.
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

    # Track available GPUs and unprocessed data
    available_gpus = set(range(num_gpus))
    unprocessed_data = all_data.copy()
    all_results = []
    disable_talker = True

    while unprocessed_data and available_gpus:
        # Distribute remaining data among available GPUs
        chunks = [[] for _ in range(len(available_gpus))]
        gpu_list = list(available_gpus)
        
        for i, data_item in enumerate(unprocessed_data):
            chunks[i % len(gpu_list)].append(data_item)

        futures = {}
        unprocessed_data = []  # Reset for collecting failed items

        with ProcessPoolExecutor(max_workers=len(available_gpus)) as executor:
            # Submit chunks to available GPUs
            for i, chunk in enumerate(chunks):
                if chunk:  # Only submit non-empty chunks
                    gpu_id = gpu_list[i]
                    futures[executor.submit(process_chunk_on_gpu, chunk, gpu_id, 
                                         args.model_path, args.data_dir, disable_talker)] = gpu_id

            main_progress = tqdm(
                as_completed(futures), 
                total=len(futures), 
                desc="Overall Progress", 
                position=0,
            )

            for future in main_progress:
                try:
                    result = future.result()
                    if result["status"] == "OOM":
                        available_gpus.remove(result["device_id"])
                        all_results.extend(result["processed_results"])
                        original_chunk = chunks[gpu_list.index(result["device_id"])]
                        processed_indices = {item['_original_index'] for item in result["processed_results"]}
                        unprocessed_data.extend(item for item in original_chunk 
                                              if item['_original_index'] not in processed_indices)
                        tqdm.write(f"GPU {result['device_id']} removed from pool. {len(available_gpus)} GPUs remaining.")
                    else:
                        all_results.extend(result["processed_results"])
                        
                except Exception as e:
                    gpu_id = futures[future]
                    tqdm.write(f"Unexpected error on GPU {gpu_id}: {e}")
                    available_gpus.remove(gpu_id)

        if unprocessed_data and not available_gpus:
            print("No GPUs remaining to process data. Terminating.")
            break

    # Sort and save results
    all_results.sort(key=lambda x: x.get('_original_index', 0))
    for item in all_results:
        if '_original_index' in item:
            del item['_original_index']

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"\nProcessed {len(all_results)} of {len(all_data)} items. Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen-Omni-7b inference in parallel on multiple GPUs.")
    parser.add_argument("--input_meta_file", type=str, default="/mnt/data/panleyi/panleyi-data/Omni-SafetyBench-new/meta_files/unimodal/video-only.jsonl", help="Input JSONL file")
    parser.add_argument("--output_file", type=str, default="/path/to/output_file", help="Output JSONL file")
    parser.add_argument("--data_dir", type=str, default="/mnt/data/panleyi/panleyi-data/Omni-SafetyBench-new/", help="Data directory")
    parser.add_argument("--model_path", type=str, default="/mnt/data_cpfs/panleyi/Qwen2.5-Omni-7B", help="Path to the Qwen-Omni model")
    args = parser.parse_args()
    main(args)