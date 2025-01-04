import os
import json
from pathlib import Path


def read_page_content(file_path):
    """Read content from a markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return ""


def process_pages_to_json(hash_directory):
    """Process pages in a hash directory and create JSON structure."""
    try:
        # Get all markdown files and sort them by page number
        md_files = sorted(
            Path(hash_directory).glob("page*.md"),
            key=lambda x: int(x.stem.replace('page', ''))
        )

        total_pages = len(md_files)
        if total_pages == 0:
            print(f"No markdown files found in {hash_directory}")
            return

        # Calculate number of complete groups (floor division)
        group_size = 50
        num_groups = total_pages // group_size
        if total_pages % group_size != 0:
            # If there are remaining pages, we'll add them to the last group
            num_groups = max(1, num_groups)  # Ensure at least one group

        json_structure = []

        # Process each group
        for group_num in range(num_groups):
            start_idx = group_num * group_size
            # For the last group, include all remaining pages
            end_idx = total_pages if group_num == num_groups - 1 else (group_num + 1) * group_size

            # Create group object
            group_obj = {
                "id": group_num + 1,
                "text": "",  # Will be filled with combined text
                "chunks": []
            }

            group_text = []

            # Process each page in the group
            for idx in range(start_idx, end_idx):
                page_content = read_page_content(md_files[idx])

                # Add to chunks
                chunk_obj = {
                    "id": idx + 1,
                    "text": page_content
                }
                group_obj["chunks"].append(chunk_obj)

                # Add to combined text
                group_text.append(page_content)

            # Set combined text for the group
            group_obj["text"] = "\n".join(group_text)

            json_structure.append(group_obj)

        # Save JSON file in the hash directory
        output_file = Path(hash_directory) / "grouped_pages.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_structure, f, indent=2, ensure_ascii=False)

        print(f"Created JSON file: {output_file}")
        print(f"Total pages: {total_pages}")
        print(f"Number of groups: {num_groups}")
        if total_pages % group_size != 0:
            print(f"Last group contains {total_pages - ((num_groups - 1) * group_size)} pages")

        return output_file

    except Exception as e:
        print(f"Error processing directory {hash_directory}: {str(e)}")
        return None


def process_all_hash_directories(processed_dir):
    """Process all hash directories in the processed directory."""
    processed_dir = Path(processed_dir)

    if not processed_dir.exists():
        print(f"Processed directory not found: {processed_dir}")
        return

    # Process each hash directory
    for hash_dir in processed_dir.iterdir():
        if hash_dir.is_dir():
            print(f"\nProcessing directory: {hash_dir}")
            process_pages_to_json(hash_dir)


def main():
    # Define directory
    base_dir = "../DOCS"
    processed_dir = os.path.join(base_dir, "processed")

    print("Starting page grouping process...")
    process_all_hash_directories(processed_dir)
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
