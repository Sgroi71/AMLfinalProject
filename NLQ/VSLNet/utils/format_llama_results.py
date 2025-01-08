import argparse
import json
import os
import random

def format_llama_results_for_vslnet(llama_results, output_dir, split_ratios=(0.8, 0.1, 0.1)):
    """
    Format LLaMA-generated results for VSLNet pretraining and fine-tuning.
    
    Args:
    - llama_results (dict): Dictionary with LLaMA-generated results.
    - output_dir (str): Directory to save formatted JSON files.
    - split_ratios (tuple): Ratios for splitting data into train, val, and test sets.
    
    Returns:
    - None
    """
    # Split data into train, val, and test sets
    video_uids = list(llama_results.keys())
    random.shuffle(video_uids)
    
    num_train = int(len(video_uids) * split_ratios[0])
    num_val = int(len(video_uids) * split_ratios[1])
    
    train_set = video_uids[:num_train]
    val_set = video_uids[num_train:num_train + num_val]
    test_set = video_uids[num_train + num_val:]
    
    splits = {"train": train_set, "val": val_set, "test": test_set}
    
    formatted_data = {"version": "1.0", "date": "2024-01-01", "description": "Pretraining data generated by LLaMA", "videos": []}
    
    for split, video_list in splits.items():
        for video_uid in video_list:
            if video_uid not in llama_results:
                continue
            
            video_entry = {"video_uid": video_uid, "clips": [], "split": split}
            
            for narration_entry in llama_results[video_uid]:
                video_entry["clips"].append({
                    "clip_uid": None,
                    "video_start_sec": narration_entry["start_sec"],
                    "video_end_sec": narration_entry["end_sec"],
                    "clip_start_sec": narration_entry["start_sec"],
                    "clip_end_sec": narration_entry["end_sec"],
                    "source_clip_uid": None,
                    "annotations": [
                        {
                            "language_queries": [
                                {
                                    "query": narration_entry["query"],
                                    "start_sec": narration_entry["start_sec"],
                                    "end_sec": narration_entry["end_sec"]
                                }
                            ]
                        }
                    ]
                })
            
            formatted_data["videos"].append(video_entry)
    
    # Save formatted data to JSON files
    os.makedirs(output_dir, exist_ok=True)
    for split in splits:
        split_data = {
            "version": "1.0",
            "date": "2025-01-07",
            "description": f"{split.capitalize()} data for VSLNet",
            "videos": [v for v in formatted_data["videos"] if v["split"] == split]
        }
        output_path = os.path.join(output_dir, f"{split}_formatted.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=4)
    
    print(f"Formatted data saved to {output_dir}")


def main(args):
    with open(args.llama_results_path, "r") as file:
        llama_results = json.load(file)

    format_llama_results_for_vslnet(llama_results, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format the results for the VSLNet.")
    parser.add_argument("llama_results_path", help="Path to LLaMA-generated results")
    parser.add_argument("output_dir", help="Output directory for formatted JSON files")

    args = parser.parse_args()
    main(args)