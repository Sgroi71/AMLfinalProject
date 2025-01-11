import random
import json
import os
import options
import torch
from llama_oracle import interact_with_llama
from utils.preprocess_generated_data import remove_videos_with_no_questions_or_answers
from utils.format_llama_results import format_llama_results_for_vslnet

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

def load_omnivore_features(omnivore_features_path):
    """Load omnivore features and organize it by video_uid for quick lookup."""
    # omnivore features are stored as .pt in a folder and each feature filename corresponds to a videouid
    
    omnivore_features = {}
    for filename in os.listdir(omnivore_features_path):
        if filename.endswith(".pt"):
            video_uid = filename.split(".")[0]
            omnivore_features[video_uid] = ""
    return omnivore_features

def generate_random_data(filtered_narration, clips_by_video,Nnar, NConsecutivenar,features,omnivore_features_path):
    if features == "omnivore":
        omnivore_features = load_omnivore_features(omnivore_features_path)
    result = {}

    # Select Nvid random videos
    video_keys = filtered_narration.keys()

    for video_key in video_keys:
        
        if video_key not in clips_by_video or (features == "omnivore" and video_key not in omnivore_features):
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

            query_start_sec = consecutive[0]["timestamp_sec"]
            query_end_sec = consecutive[-1]["timestamp_sec"]

            # Check if a corresponding clip exists in the clips dictionary
            matching_clip = next(
                (clip for clip in available_clips if clip["video_start_sec"] <= query_start_sec <= clip["video_end_sec"]), 
                None
            )
            if not matching_clip:
                continue  # Skip if no matching clip found

            
            # Compute start and end frames using fps
            fps = matching_clip["clip_metadata"]["fps"]
            clip_start_frame = int(round(fps * query_start_sec))
            clip_end_frame = int(round(fps * query_end_sec))

            video_start_sec = matching_clip["video_start_sec"]
            video_end_sec = matching_clip["video_end_sec"]

            narration_texts = [n["narration_text"] for n in consecutive if "unsure" not in n["narration_text"] and "Unsure" not in n["narration_text"] and "UNSURE" not in n["narration_text"]]
            selected_narrations.append({
                "video_start_sec": video_start_sec,
                "video_end_sec": video_end_sec,
                "query_start_sec": query_start_sec,
                "query_end_sec": query_end_sec,
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
    configs, parser = options.read_command_line()
    
    narration_data = json.load(open(configs.narration_data))
    nlq_train = json.load(open(configs.nlq_train))
    nlq_val = json.load(open(configs.nlq_val))
    nlq_test = json.load(open(configs.nlq_test))
    clips_by_video = load_clips_dict("/content/AMLfinalProject/NLQ/VSLNet/jsons/clips_dict.json")

    train_video_uids = {video['video_uid'] for video in nlq_train['videos']}
    val_video_uids = {video['video_uid'] for video in nlq_val['videos']}
    test_video_uids = {video['video_uid'] for video in nlq_test['videos']}
    
    exclude_uids = train_video_uids | val_video_uids | test_video_uids
    if configs.features == "omnivore":
        filtered_narration=narration_data
    else:
        filtered_narration = {uid: entry for uid, entry in narration_data.items() if uid not in exclude_uids}
    
    random_data = generate_random_data(filtered_narration, clips_by_video,Nnar, NConsecutivenar,configs.features,configs.feature_dir)
    llama_output=interact_with_llama(random_data)
    filtered_data=remove_videos_with_no_questions_or_answers(llama_output)
    # the following function will save the data in the output_dir
    format_llama_results_for_vslnet(filtered_data, configs.output_dir)

