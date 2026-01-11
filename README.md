# ParlaMint NLP Analysis

This project implements an NLP pipeline for analyzing parliamentary proceedings from the [ParlaMint dataset](https://github.com/CLARIN-ERIC/ParlaMint). It focuses on extracting, cleaning, and analyzing speeches to uncover patterns in sentiment and political alignment (Government vs. Opposition).

## Features

- **Data Ingestion**: Automated downloading and parsing of TEI-formatted XML files.
- **Metadata Enrichment**:
  - Speaker demographics (Name, Sex).
  - Temporal party affiliation mapping.
  - Gov/Opp classification based on parliamentary coalition data.
- **Analysis**:
  - Sentiment analysis of speeches.
  - Topic modelling and visualization.

## Getting Started

### Prerequisites

- Python 3.12+
- Jupyter Notebook
- Dependencies listed in `requirements.txt`

### Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Data**:
   Run the included script to clone the ParlaMint repository.
   ```bash
   chmod +x download_data.sh
   ./download_data.sh
   ```

## Project Structure

- `download_data.sh`: Script to fetch the ParlaMint dataset.
- `archive/exploration.ipynb`: Initial data exploration and prototyping.
- `data.py`: Reusable Python module containing core parsing, taxonomy loading, and metadata extraction logic.
- `ParlaMint/`: Data directory (created after running setup).