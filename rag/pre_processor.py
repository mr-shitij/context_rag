# preprocessor.py

import os
import hashlib
import shutil
import json
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader


class PDFPreProcessor:
    def __init__(self, raw_dir: str, processed_dir: str, group_size: int = 50):
        """
        Initialize the pre-processor with paths for raw and processed PDFs,
        and the desired group size (number of pages per group).
        """
        load_dotenv()  # load environment variables from .env file
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.group_size = group_size
        self.parser = self.setup_pdf_parser()

    def setup_pdf_parser(self) -> LlamaParse:
        """Set up the PDF parser using LlamaParse and environment configurations."""
        api_key = os.getenv('LLAMA_CLOUD_API_KEY')
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY not found in environment variables")
        # Initialize LlamaParse with result_type "markdown"
        return LlamaParse(result_type="markdown", api_key=api_key)

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file for deduplication."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def parse_pdf(self, file_path: Path, output_dir: Path):
        """
        Parse the PDF file at 'file_path' and save each page as a markdown file
        into the 'output_dir'.
        """
        file_extractor = {".pdf": self.parser}
        documents = SimpleDirectoryReader(
            input_files=[str(file_path)],
            file_extractor=file_extractor
        ).load_data()
        for idx, doc in enumerate(documents, 1):
            page_file = output_dir / f"page{idx}.md"
            with open(page_file, 'w', encoding='utf-8') as f:
                f.write(doc.text)
            print(f"Created: {page_file}")

    def process_pdfs(self):
        """
        Process all PDF files in the raw directory. For each unique PDF (determined by its hash),
        create a folder under the processed directory, copy the PDF there, and parse it into markdown pages.
        Duplicate PDFs are skipped.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        hash_map = {}
        duplicate_count = 0

        for file_path in self.raw_dir.glob("**/*.pdf"):
            try:
                file_hash = self.calculate_file_hash(file_path)
                if file_hash not in hash_map:
                    hash_map[file_hash] = file_path
                    hash_dir = self.processed_dir / file_hash
                    hash_dir.mkdir(exist_ok=True)
                    pdf_destination = hash_dir / file_path.name
                    shutil.copy2(file_path, pdf_destination)
                    print(f"Parsing PDF: {file_path.name}")
                    self.parse_pdf(file_path, hash_dir)
                    print(f"Processed: {file_path.name} -> {file_hash}")
                else:
                    duplicate_count += 1
                    print(f"Duplicate found: {file_path.name} (same as {hash_map[file_hash].name})")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

        print(f"\nProcessing complete:")
        print(f"Unique files: {len(hash_map)}")
        print(f"Duplicates found: {duplicate_count}")

    def read_page_content(self, file_path: Path) -> str:
        """Read the full text from a markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return ""

    def process_pages_to_json(self, hash_directory: Path) -> Path:
        """
        Process all markdown page files in the given hash directory, grouping pages
        into groups of size 'group_size'. Each group (document) will have:
          - an 'id'
          - a 'text' field containing combined text of all pages in the group
          - a 'chunks' array, where each chunk represents one page with its 'id' and 'content'
        The resulting JSON structure is written to a file named 'grouped_pages.json'
        in the same hash directory.
        """
        md_files = sorted(
            hash_directory.glob("page*.md"),
            key=lambda x: int(x.stem.replace('page', ''))
        )
        total_pages = len(md_files)
        if total_pages == 0:
            print(f"No markdown files found in {hash_directory}")
            return None

        num_groups = total_pages // self.group_size
        if total_pages % self.group_size != 0:
            num_groups = max(1, num_groups)

        json_structure = []
        for group_num in range(num_groups):
            start_idx = group_num * self.group_size
            end_idx = total_pages if group_num == num_groups - 1 else (group_num + 1) * self.group_size
            group_obj = {
                "id": group_num + 1,
                "text": "",
                "chunks": []
            }
            group_text = []
            for idx in range(start_idx, end_idx):
                page_content = self.read_page_content(md_files[idx])
                chunk_obj = {
                    "id": idx + 1,
                    "content": page_content
                }
                group_obj["chunks"].append(chunk_obj)
                group_text.append(page_content)
            group_obj["text"] = "\n".join(group_text)
            json_structure.append(group_obj)

        output_file = hash_directory / "grouped_pages.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_structure, f, indent=2, ensure_ascii=False)

        print(f"Created JSON file: {output_file}")
        print(f"Total pages: {total_pages}")
        print(f"Number of groups: {num_groups}")
        if total_pages % self.group_size != 0:
            print(f"Last group contains {total_pages - ((num_groups - 1) * self.group_size)} pages")
        return output_file

    def process_all_hash_directories(self):
        """Process every hash directory under the processed directory to create a grouped JSON file."""
        if not self.processed_dir.exists():
            print(f"Processed directory not found: {self.processed_dir}")
            return

        for hash_dir in self.processed_dir.iterdir():
            if hash_dir.is_dir():
                print(f"\nProcessing directory: {hash_dir}")
                self.process_pages_to_json(hash_dir)

    def process(self):
        print("Starting PDF processing...")
        self.process_pdfs()
        print("Starting page grouping process...")
        self.process_all_hash_directories()
        print("Processing complete!")


def main():
    base_dir = "../DOCS"
    raw_dir = os.path.join(base_dir, "raw")
    processed_dir = os.path.join(base_dir, "processed")

    preprocessor = PDFPreProcessor(raw_dir, processed_dir, group_size=50)
    preprocessor.process()


if __name__ == "__main__":
    main()
