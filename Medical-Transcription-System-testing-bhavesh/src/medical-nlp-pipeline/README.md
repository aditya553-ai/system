# Medical NLP Pipeline

This project is a comprehensive Medical NLP Pipeline designed to recognize medical entities, extract relationships, and construct knowledge graphs from medical transcripts. It integrates various components, including entity recognition, terminology resolution, relation extraction, and knowledge graph construction, utilizing state-of-the-art techniques and models.

## Features

- **Entity Recognition**: Identifies medical entities in transcripts using spaCy and custom patterns.
- **Hybrid Entity Mapping**: Combines open-source terminology resolvers, LLM-guided mapping, and multi-agent cross-verification for accurate entity normalization.
- **Relation Extraction**: Extracts semantic relationships between recognized entities to understand their interactions.
- **Knowledge Graph Construction**: Builds a context-aware knowledge graph from extracted entities and relationships, enabling advanced querying and analysis.
- **Error Correction**: Detects and corrects transcription errors to improve data quality.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd medical-nlp-pipeline
pip install -r requirements.txt
```

## Usage

The main entry point for the pipeline is `pipeline.py`. You can run the pipeline with the following command:

```bash
python src/pipeline.py path/to/transcript.json
```

### Example

To see the pipeline in action, refer to the Jupyter notebook located in the `notebooks` directory:

```bash
notebooks/pipeline_demo.ipynb
```

## Testing

Unit tests are provided for each component of the pipeline. To run the tests, use:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.