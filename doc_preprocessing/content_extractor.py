import os
import hashlib
import shutil
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader


def setup_pdf_parser():
    """Set up the PDF parser with LlamaParse and environment configurations."""
    try:
        # Load environment variables
        load_dotenv()

        # Verify API key
        api_key = os.getenv('LLAMA_CLOUD_API_KEY')
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY not found in environment variables")

        # Initialize LlamaParse
        return LlamaParse(result_type="markdown", api_key=api_key)

    except Exception as e:
        print(f"Error setting up PDF parser: {str(e)}")
        raise


def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def parse_pdf(file_path, output_dir):
    """Parse PDF and save each page as markdown."""
    try:
        parser = setup_pdf_parser()
        file_extractor = {".pdf": parser}

        # Parse the PDF
        documents = SimpleDirectoryReader(
            input_files=[str(file_path)],
            file_extractor=file_extractor
        ).load_data()

        # Save each page as separate markdown file
        for idx, doc in enumerate(documents, 1):
            page_file = output_dir / f"page{idx}.md"
            with open(page_file, 'w', encoding='utf-8') as f:
                f.write(doc.text)
            print(f"Created: {page_file}")

    except Exception as e:
        print(f"Error parsing PDF {file_path}: {str(e)}")
        raise


def process_pdfs(raw_dir, processed_dir):
    """Process PDFs from raw directory and organize them by hash with parsed content."""
    # Create processed directory if it doesn't exist
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    # Dictionary to store hash-to-file mapping
    hash_map = {}
    duplicate_count = 0

    # Process all PDF files in raw directory
    for file_path in Path(raw_dir).glob("**/*.pdf"):
        try:
            # Calculate hash
            file_hash = calculate_file_hash(file_path)

            if file_hash not in hash_map:
                # This is a unique file
                hash_map[file_hash] = file_path

                # Create directory for this hash
                hash_dir = Path(processed_dir) / file_hash
                hash_dir.mkdir(exist_ok=True)

                # Copy original PDF
                pdf_destination = hash_dir / file_path.name
                shutil.copy2(file_path, pdf_destination)

                # Parse PDF and save pages
                print(f"Parsing PDF: {file_path.name}")
                parse_pdf(file_path, hash_dir)

                print(f"Processed: {file_path.name} -> {file_hash}")
            else:
                # This is a duplicate
                duplicate_count += 1
                print(f"Duplicate found: {file_path.name} (same as {hash_map[file_hash].name})")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    print(f"\nProcessing complete:")
    print(f"Unique files: {len(hash_map)}")
    print(f"Duplicates found: {duplicate_count}")


def main():
    # Define directories
    base_dir = "../DOCS"
    raw_dir = os.path.join(base_dir, "raw")
    processed_dir = os.path.join(base_dir, "processed")

    # Check if raw directory exists
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory not found at {raw_dir}")
        return

    print("Starting PDF processing...")
    process_pdfs(raw_dir, processed_dir)


if __name__ == "__main__":
    main()
