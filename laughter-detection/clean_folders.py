import os
import shutil

def delete_files_in_directory(directory):
    try:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print(f"All files in {directory} have been deleted.")
        else:
            print(f"Directory {directory} does not exist.")
    except Exception as e:
        print(f"Failed to delete files in {directory} because: {e}")

def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} has been deleted.")
        else:
            print(f"File {file_path} does not exist.")
    except Exception as e:
        print(f"Failed to delete file {file_path} because: {e}")

def main():
    # Directories to clear
    directories = [
        '/disk/data/podcast_audio/v_laughter/audio_files',
        '/disk/data/podcast_audio/v_laughter/chunks'
    ]
    # Specific file to delete
    file_to_delete = '/disk/data/podcast_audio/v_laughter/timestamps.txt'
    
    # Delete files in directories
    for directory in directories:
        delete_files_in_directory(directory)
    
    # Delete the specific file
    delete_file(file_to_delete)

if __name__ == "__main__":
    main()
