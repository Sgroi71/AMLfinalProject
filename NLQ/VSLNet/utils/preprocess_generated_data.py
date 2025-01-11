import json
import argparse


def remove_videos_with_no_questions_or_answers(data):
    """
    Reads the JSON from `input_path`, and writes a new JSON to `output_path`
    after removing all videos that contain no questions or no answers in their clips.
    """

    # Filter the data to include only videos with at least one question and one answer
    filtered_data = {
        video_uid: [
            clip for clip in clips
            if len(clip["questions"]) > 0 and len(clip["answers"]) > 0
        ]
        for video_uid, clips in data.items()
    }

    # Remove videos with no valid clips remaining
    filtered_data = {video_uid: clips for video_uid, clips in filtered_data.items() if clips}

    return filtered_data



