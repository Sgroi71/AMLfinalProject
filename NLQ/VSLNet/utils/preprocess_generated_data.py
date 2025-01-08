import json
import argparse


def remove_videos_with_no_questions_or_answers(input_path, output_path):
    """
    Reads the JSON from `input_path`, and writes a new JSON to `output_path`
    after removing all videos that contain no questions or no answers in their clips.
    """
    # Load the JSON
    with open(input_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

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

    # Write the filtered data to a file
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(filtered_data, outfile, indent=4)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Remove videos with no questions or answers from JSON data.")
    parser.add_argument(
        "--narration_filename", 
        type=str, 
        required=True, 
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Path to save the output JSON file."
    )

    args = parser.parse_args()

    # Call the function with the parsed arguments
    remove_videos_with_no_questions_or_answers(args.narration_filename, args.output_dir)