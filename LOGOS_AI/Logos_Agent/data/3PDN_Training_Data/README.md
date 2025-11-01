# 3PDN Training Data Index
# ========================

Generated: 2025-10-30T20:40:12.370842
Source: Three_Pillars_of_Divine_Necessity

## Directory Structure

- **extracted_documents/**: Text extracted from DOCX and PDF files
- **original_text_files/**: Original .txt files copied directly  
- **markdown_docs/**: Original .md files copied directly
- **coq_proofs/**: Coq verification files (.v)
- **isabelle_theories/**: Isabelle/HOL theory files (.thy)
- **metadata/**: Extraction reports and metadata

## Extraction Summary

Total files processed: 79
Successful extractions: 78
Failed extractions: 1

## Files by Type
- .docx: 30 files
- .zip: 3 files
- .txt: 1 files
- .pdf: 23 files
- .thy: 6 files
- .v: 6 files
- .md: 10 files

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
