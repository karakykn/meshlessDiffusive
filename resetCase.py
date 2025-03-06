import os
import shutil

# Define the base directory where caseName is located
base_dir = 'syntheticNetwork'  # Replace with the actual path to the caseName directory

# Walk through the directories
for segment_dir in os.listdir(base_dir):
    segment_path = os.path.join(base_dir, segment_dir)

    # Make sure we only process directories
    if os.path.isdir(segment_path) and segment_dir.startswith('segment'):
        run_dir = os.path.join(segment_path, 'run')

        # Check if the 'run' directory exists
        if os.path.isdir(run_dir):
            for time_dir in os.listdir(run_dir):
                time_dir_path = os.path.join(run_dir, time_dir)

                # Skip if it's the '0' directory
                if os.path.isdir(time_dir_path) and time_dir != '0':
                    shutil.rmtree(time_dir_path)  # Remove the directory

print("Completed cleaning up the 'run' directories.")