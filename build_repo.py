import os
import re

def build_repo(source_file):
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find file blocks based on the prompt's format:
    # File: path/to/filename.ext
    # ```language
    # code
    # ```
    # Handle both `backtick` and plain formats, and escaped underscores
    pattern = r"File:\s*`?([\w.\\/_ -]+)`?\n\n```[\w]*\n(.*?)```"

    matches = re.findall(pattern, content, re.DOTALL)

    # Clean up escaped underscores in file paths (e.g., config\_parser.py -> config_parser.py)
    matches = [(path.replace('\\_', '_'), code) for path, code in matches]

    if not matches:
        print("No files found! Make sure the Gemini output format matches: **File: path/to/file**")
        return

    print(f"Found {len(matches)} files to create...")

    for file_path, file_content in matches:
        # Create directories if they don't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        # Write the file
        with open(file_path, 'w', encoding='utf-8') as out_f:
            out_f.write(file_content.strip())
        
        print(f"Written: {file_path}")

    print("\nRepo build complete! 🚀")

if __name__ == "__main__":
    # Create a dummy file if it doesn't exist so the user knows where to paste
    if not os.path.exists("gemini_output.txt"):
        with open("gemini_output.txt", "w") as f:
            f.write("[PASTE THE FULL GEMINI CHAT OUTPUT HERE]")
        print("Created 'gemini_output.txt'. Paste the AI response in there and run this script again.")
    else:
        build_repo("gemini_output.txt")