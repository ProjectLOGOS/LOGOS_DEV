#!/usr/bin/env python3
"""
3PDN Content Extractor for Git Integration
==========================================

This script extracts text content from binary files in Three_Pillars_of_Divine_Necessity
and converts them to git-trackable formats for LOGOS training data.
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThreePDNExtractor:
    """Extract and prepare 3PDN content for git tracking"""
    
    def __init__(self, source_dir="Three_Pillars_of_Divine_Necessity", target_dir="3PDN_Training_Data"):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.extraction_report = {
            "extraction_date": datetime.now().isoformat(),
            "total_files_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "files_by_type": {},
            "extraction_log": []
        }
    
    def setup_target_directory(self):
        """Create organized target directory structure"""
        
        # Create main directories
        directories = [
            "extracted_documents",
            "original_text_files", 
            "coq_proofs",
            "isabelle_theories",
            "markdown_docs",
            "metadata"
        ]
        
        for dir_name in directories:
            (self.target_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created target directory structure: {self.target_dir}")
    
    def extract_text_from_docx(self, docx_path):
        """Extract text from .docx files using python-docx if available"""
        
        try:
            # Try to use python-docx
            import docx
            
            doc = docx.Document(docx_path)
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text.strip())
            
            return "\n\n".join(text_content)
            
        except ImportError:
            logger.warning("python-docx not available, creating placeholder for: " + str(docx_path))
            return f"""# {docx_path.name}

This document was extracted from: {docx_path}
Original format: Microsoft Word Document (.docx)

[EXTRACTION PLACEHOLDER - Install python-docx to extract full content]

To extract manually:
1. Install python-docx: pip install python-docx
2. Re-run this extraction script
3. Or manually convert to .txt and place in original_text_files/

File size: {docx_path.stat().st_size} bytes
Modified: {datetime.fromtimestamp(docx_path.stat().st_mtime)}
"""
        except Exception as e:
            logger.error(f"Failed to extract from {docx_path}: {e}")
            return f"# Extraction Failed\n\nError extracting from: {docx_path}\nError: {str(e)}"
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF files using PyPDF2 if available"""
        
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                
                return "\n\n".join(text_content)
                
        except ImportError:
            logger.warning("PyPDF2 not available, creating placeholder for: " + str(pdf_path))
            return f"""# {pdf_path.name}

This document was extracted from: {pdf_path}
Original format: Portable Document Format (.pdf)

[EXTRACTION PLACEHOLDER - Install PyPDF2 to extract full content]

To extract manually:
1. Install PyPDF2: pip install PyPDF2
2. Re-run this extraction script  
3. Or manually convert to .txt and place in original_text_files/

File size: {pdf_path.stat().st_size} bytes
Modified: {datetime.fromtimestamp(pdf_path.stat().st_mtime)}
"""
        except Exception as e:
            logger.error(f"Failed to extract from {pdf_path}: {e}")
            return f"# Extraction Failed\n\nError extracting from: {pdf_path}\nError: {str(e)}"
    
    def copy_text_files(self, source_path, target_subdir):
        """Copy existing text-based files directly"""
        
        target_path = self.target_dir / target_subdir / source_path.name
        
        try:
            shutil.copy2(source_path, target_path)
            logger.info(f"Copied: {source_path} -> {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy {source_path}: {e}")
            return False
    
    def process_file(self, file_path):
        """Process individual files based on their type"""
        
        relative_path = file_path.relative_to(self.source_dir)
        self.extraction_report["total_files_processed"] += 1
        
        # Track file types
        ext = file_path.suffix.lower()
        if ext not in self.extraction_report["files_by_type"]:
            self.extraction_report["files_by_type"][ext] = 0
        self.extraction_report["files_by_type"][ext] += 1
        
        try:
            if ext == ".docx":
                # Extract text from Word documents
                extracted_text = self.extract_text_from_docx(file_path)
                
                # Save as markdown
                output_path = self.target_dir / "extracted_documents" / f"{file_path.stem}.md"
                output_path.write_text(extracted_text, encoding='utf-8')
                
                logger.info(f"Extracted DOCX: {relative_path}")
                
            elif ext == ".pdf":
                # Extract text from PDFs
                extracted_text = self.extract_text_from_pdf(file_path)
                
                # Save as markdown
                output_path = self.target_dir / "extracted_documents" / f"{file_path.stem}.md"
                output_path.write_text(extracted_text, encoding='utf-8')
                
                logger.info(f"Extracted PDF: {relative_path}")
                
            elif ext in [".txt", ".md"]:
                # Copy text files directly
                target_subdir = "markdown_docs" if ext == ".md" else "original_text_files"
                success = self.copy_text_files(file_path, target_subdir)
                if not success:
                    raise Exception("Copy failed")
                
            elif ext == ".v":
                # Copy Coq proof files
                success = self.copy_text_files(file_path, "coq_proofs")
                if not success:
                    raise Exception("Copy failed")
                
            elif ext == ".thy":
                # Copy Isabelle theory files
                success = self.copy_text_files(file_path, "isabelle_theories")
                if not success:
                    raise Exception("Copy failed")
                
            elif ext == ".zip":
                # Log zip files but don't extract (too complex for this script)
                logger.info(f"Skipped ZIP file: {relative_path} (manual extraction required)")
                
                # Create placeholder
                output_path = self.target_dir / "extracted_documents" / f"{file_path.stem}_ZIP_PLACEHOLDER.md"
                placeholder_text = f"""# {file_path.name}

This is a placeholder for ZIP archive: {file_path}

**Manual extraction required:**
1. Extract the ZIP archive manually
2. Process individual files using this script
3. Add extracted content to appropriate directories

File size: {file_path.stat().st_size} bytes
Modified: {datetime.fromtimestamp(file_path.stat().st_mtime)}
"""
                output_path.write_text(placeholder_text, encoding='utf-8')
                
            else:
                # Handle other files
                logger.info(f"Unknown file type: {relative_path}")
                
                # Create placeholder for unknown types
                output_path = self.target_dir / "extracted_documents" / f"{file_path.stem}_UNKNOWN.md"
                placeholder_text = f"""# {file_path.name}

Unknown file type: {ext}
Original path: {file_path}

File size: {file_path.stat().st_size} bytes
Modified: {datetime.fromtimestamp(file_path.stat().st_mtime)}

Manual processing may be required.
"""
                output_path.write_text(placeholder_text, encoding='utf-8')
            
            self.extraction_report["successful_extractions"] += 1
            self.extraction_report["extraction_log"].append({
                "file": str(relative_path),
                "type": ext,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to process {relative_path}: {e}")
            self.extraction_report["failed_extractions"] += 1
            self.extraction_report["extraction_log"].append({
                "file": str(relative_path),
                "type": ext,
                "status": "failed", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    def create_training_data_index(self):
        """Create an index file for LOGOS training data"""
        
        index_content = f"""# 3PDN Training Data Index
# ========================

Generated: {datetime.now().isoformat()}
Source: Three_Pillars_of_Divine_Necessity

## Directory Structure

- **extracted_documents/**: Text extracted from DOCX and PDF files
- **original_text_files/**: Original .txt files copied directly  
- **markdown_docs/**: Original .md files copied directly
- **coq_proofs/**: Coq verification files (.v)
- **isabelle_theories/**: Isabelle/HOL theory files (.thy)
- **metadata/**: Extraction reports and metadata

## Extraction Summary

Total files processed: {self.extraction_report['total_files_processed']}
Successful extractions: {self.extraction_report['successful_extractions']}
Failed extractions: {self.extraction_report['failed_extractions']}

## Files by Type
"""

        for file_type, count in self.extraction_report["files_by_type"].items():
            index_content += f"- {file_type or '(no extension)'}: {count} files\n"

        index_content += """
## Usage for LOGOS Training

This directory contains training data for LOGOS AI system initialization:

1. **Core Arguments**: Extracted philosophical and logical arguments
2. **Formal Proofs**: Coq and Isabelle verification files  
3. **Foundational Texts**: Base texts for Trinity logic understanding
4. **Meta-logical Framework**: Supporting mathematical structures

## Integration with LOGOS

Add to LOGOS startup configuration:
```python
training_data_paths = [
    "3PDN_Training_Data/extracted_documents/",
    "3PDN_Training_Data/original_text_files/", 
    "3PDN_Training_Data/coq_proofs/"
]
```
"""
        
        # Save index
        index_path = self.target_dir / "README.md"
        index_path.write_text(index_content, encoding='utf-8')
        
        # Save detailed metadata
        metadata_path = self.target_dir / "metadata" / "extraction_report.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.extraction_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created training data index: {index_path}")
        logger.info(f"Saved extraction report: {metadata_path}")
    
    def extract_all(self):
        """Main extraction process"""
        
        if not self.source_dir.exists():
            logger.error(f"Source directory not found: {self.source_dir}")
            return False
        
        logger.info(f"Starting 3PDN extraction from: {self.source_dir}")
        
        # Setup target directory
        self.setup_target_directory()
        
        # Process all files recursively
        for file_path in self.source_dir.rglob("*"):
            if file_path.is_file():
                self.process_file(file_path)
        
        # Create training data index
        self.create_training_data_index()
        
        # Print summary
        logger.info("=" * 50)
        logger.info("3PDN EXTRACTION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Total files processed: {self.extraction_report['total_files_processed']}")
        logger.info(f"Successful extractions: {self.extraction_report['successful_extractions']}")
        logger.info(f"Failed extractions: {self.extraction_report['failed_extractions']}")
        logger.info(f"Output directory: {self.target_dir}")
        
        return True


def main():
    """Main entry point"""
    
    print("3PDN Training Data Extractor for LOGOS AI")
    print("=" * 45)
    
    # Check if source directory exists
    if not Path("Three_Pillars_of_Divine_Necessity").exists():
        print("❌ Three_Pillars_of_Divine_Necessity directory not found!")
        print("   Make sure you're running this from the LOGOS_DEV root directory.")
        return 1
    
    # Create extractor and run
    extractor = ThreePDNExtractor()
    success = extractor.extract_all()
    
    if success:
        print("\n🎉 3PDN extraction completed successfully!")
        print(f"📁 Training data ready in: {extractor.target_dir}")
        print("\n📋 Next steps:")
        print("   1. Review extracted content in 3PDN_Training_Data/")
        print("   2. git add 3PDN_Training_Data/")
        print("   3. git commit -m 'feat: Add 3PDN training data for LOGOS'")
        print("   4. git push")
        return 0
    else:
        print("\n❌ 3PDN extraction failed!")
        return 1


if __name__ == "__main__":
    exit(main())