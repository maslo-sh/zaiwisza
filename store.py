from haystack import Pipeline
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.joiners import DocumentJoiner
from haystack.utils import Secret

import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "your-pinecone-api-key")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "zaiwisza-index")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_SERVERLESS_ENV", "us-east-1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-openai-api-key")

def write_document(doc_name: str):
    store = PineconeDocumentStore(
        index=PINECONE_INDEX_NAME,
        namespace="default",
        dimension=1536,
        metric="cosine",
        spec={
            "serverless": {"region": PINECONE_ENVIRONMENT, "cloud": "aws"}
        }
    )

    pdf_file_converter = PyPDFToDocument()
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter(split_by="word", split_length=200, split_overlap=50)
    embedder = OpenAIDocumentEmbedder(api_key=Secret.from_token(OPENAI_API_KEY),)
    joiner = DocumentJoiner()
    writer = DocumentWriter(store)

    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component(instance=pdf_file_converter, name="pypdf_converter")
    preprocessing_pipeline.add_component(instance=joiner, name="document_joiner")
    preprocessing_pipeline.add_component(instance=cleaner, name="document_cleaner")
    preprocessing_pipeline.add_component(instance=splitter, name="document_splitter")
    preprocessing_pipeline.add_component(instance=embedder, name="document_embedder")
    preprocessing_pipeline.add_component(instance=writer, name="document_writer")

    preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
    preprocessing_pipeline.connect("document_joiner", "document_cleaner")
    preprocessing_pipeline.connect("document_cleaner", "document_splitter")
    preprocessing_pipeline.connect("document_splitter", "document_embedder")
    preprocessing_pipeline.connect("document_embedder", "document_writer")

    preprocessing_pipeline.run({"sources": [doc_name]})

if __name__ == "__main__":
    write_document("pdf_skauci.pdf")
