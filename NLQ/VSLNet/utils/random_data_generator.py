import random
import json

Nvid = 1700  # Number of videos to select
Nnar = 3  # Number of narrations to select per video
NConsecutivenar = 5  # Number of consecutive narrations to include

def load_clips_dict(clips_dict_path):
    """Load clips_dict.json and organize it by video_uid for quick lookup."""
    with open(clips_dict_path, "r") as f:
        clips = json.load(f)
    clips_by_video = {}
    for clip in clips:
        video_uid = clip["video_uid"]
        if video_uid not in clips_by_video:
            clips_by_video[video_uid] = []
        clips_by_video[video_uid].append(clip)
    return clips_by_video

def generate_random_data(filtered_narration, clips_by_video, Nvid, Nnar, NConsecutivenar):
    result = {}

    # Select Nvid random videos
    video_keys = random.sample(list(filtered_narration.keys()), min(Nvid, len(filtered_narration)))

    for video_key in video_keys:
        if video_key not in clips_by_video:
            continue  # Skip videos without corresponding clips

        video_data = filtered_narration[video_key]

        # Skip if 'narration_pass_1' key is missing
        if "narration_pass_1" not in video_data or "narrations" not in video_data["narration_pass_1"]:
            continue
        
        if "narration_pass_2" not in video_data:
            narrations = video_data["narration_pass_1"]["narrations"]
        else:
            narrations = random.choice([video_data["narration_pass_1"]["narrations"], video_data["narration_pass_2"]["narrations"]])
        available_clips = clips_by_video[video_key]

        if len(narrations) - 6 <= 5:
            continue

        # Select Nnar random narrations
        selected_indices = random.sample(range(len(narrations) - 6), Nnar)

        # Create the list of selected narrations
        selected_narrations = []
        for index in selected_indices:
            if index + NConsecutivenar <= len(narrations):
                consecutive = narrations[index: index + NConsecutivenar]
            else:
                consecutive = narrations[index:]

            start_sec = consecutive[0]["timestamp_sec"]
            end_sec = consecutive[-1]["timestamp_sec"]

            # Check if a corresponding clip exists in the clips dictionary
            matching_clip = next(
                (clip for clip in available_clips if clip["video_start_sec"] <= start_sec <= clip["video_end_sec"]), 
                None
            )
            if not matching_clip:
                continue  # Skip if no matching clip found

            
            # Compute start and end frames using fps
            fps = matching_clip["clip_metadata"]["fps"]
            clip_start_frame = int(round(fps * start_sec))
            clip_end_frame = int(round(fps * end_sec))

            narration_texts = [n["narration_text"] for n in consecutive if "unsure" not in n["narration_text"] and "Unsure" not in n["narration_text"] and "UNSURE" not in n["narration_text"]]
            selected_narrations.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "narrations": narration_texts,
                "clip_uid": matching_clip["clip_uid"],
                "video_start_frame": matching_clip["video_start_frame"],
                "video_end_frame": matching_clip["video_end_frame"],
                "clip_start_frame": clip_start_frame,
                "clip_end_frame": clip_end_frame
            })

        if selected_narrations:
            result[video_key] = selected_narrations

    return result

# Example usage
if __name__ == "__main__":
    narration_data = json.load(open("/home/alexhax/aml/ego4d_data/v1/annotations/narration.json"))
    nlq_train = json.load(open("/home/alexhax/aml/ego4d_data/v1/annotations/nlq_train.json"))
    nlq_val = json.load(open("/home/alexhax/aml/ego4d_data/v1/annotations/nlq_val.json"))
    nlq_test = json.load(open("/home/alexhax/aml/ego4d_data/v1/annotations/nlq_test_unannotated.json"))
    clips_by_video = load_clips_dict("/home/alexhax/aml/ego4d_data/clips_dict.json")

    train_video_uids = {video['video_uid'] for video in nlq_train['videos']}
    val_video_uids = {video['video_uid'] for video in nlq_val['videos']}
    test_video_uids = {video['video_uid'] for video in nlq_test['videos']}
    
    exclude_uids = train_video_uids | val_video_uids | test_video_uids
    filtered_narration = {uid: entry for uid, entry in narration_data.items() if uid not in exclude_uids}
    
    random_data = generate_random_data(filtered_narration, clips_by_video, Nvid, Nnar, NConsecutivenar)
    with open("random_data.json", "w") as f:
        json.dump(random_data, f, indent=4)

'''
# Save random_data into file and save path
# Convert the dictionary into a list of items (key-value pairs)
random_data = json.load(open("/content/AMLfinalProject/NLQ/VSLNet/jsons/random_data.json"))
items = list(random_data.items())

# Find the midpoint
mid = len(items) // 2

# Split the items into two halves
random_data1 = dict(items[:mid])
random_data2 = dict(items[mid:])

with open('/content/ego4d_data/v1/annotations/random_data1.json', 'w') as f:
    json.dump(random_data1, f, indent=4)
with open('/content/ego4d_data/v1/annotations/random_data2.json', 'w') as f:
    json.dump(random_data1, f, indent=4)

%%bash

source vars.sh
cd AMLfinalProject/NLQ/VSLNet
python llama_oracle.py \
    --narration_filename /content/AMLfinalProject/NLQ/VSLNet/jsons/random_data2.json \
    --output_dir /content/AMLfinalProject/NLQ/VSLNet/jsons/output_data2.json

import json

with open("/content/AMLfinalProject/NLQ/VSLNet/jsons/output_data1.json", "r") as f1:
    data1 = json.load(f1)

with open("/content/AMLfinalProject/NLQ/VSLNet/jsons/output_data2.json", "r") as f2:
    data2 = json.load(f2)

merged_data = {**data1, **data2}

with open("/content/AMLfinalProject/NLQ/VSLNet/jsons/output_data.json", "w") as f_out:
    json.dump(merged_data, f_out, indent=4)

%%bash

cd AMLfinalProject/NLQ/VSLNet/utils
python preprocess_generated_data.py \
    --input_dir /content/AMLfinalProject/NLQ/VSLNet/jsons/output_data.json \
    --output_dir /content/AMLfinalProject/NLQ/VSLNet/jsons/preprocessed_generated_data.json

%%bash

cd AMLfinalProject/NLQ/VSLNet/utils
python format_llama_results.py \
    --llama_results_path /content/AMLfinalProject/NLQ/VSLNet/jsons/preprocessed_generated_data.json \
    --output_dir /content/AMLfinalProject/NLQ/VSLNet/jsons/
'''
