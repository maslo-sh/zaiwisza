"""
Scouts d'Europe RAG Assistant

This program implements an AI assistant that:
- Uses RAG with Pinecone database as knowledge base
- Answers only questions related to topics in the knowledge base
- Uses Haystack as an AI agent framework
- Will be used by Scouts d'Europe to quickly get info about administrative tasks
"""

import os
from typing import List, Dict, Any, Optional

# Haystack imports
from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import Document
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.utils import Secret
from dotenv import load_dotenv
from haystack.components.generators.utils import print_streaming_chunk
import signal, sys

load_dotenv()

# Environment variables for API keys (should be set in your environment)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-openai-api-key")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "your-pinecone-api-key")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "zaiwisza-index")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_SERVERLESS_ENV", "us-east-1")

class ScoutsAssistant:
    def __init__(self):
        """Initialize the Scouts d'Europe RAG Assistant with Pinecone and Haystack."""
        
        # Initialize the RAG pipeline
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the Haystack RAG pipeline with Pinecone retriever."""
        cleaner = DocumentCleaner()
        splitter = DocumentSplitter()

        self.document_store = PineconeDocumentStore(
            index=PINECONE_INDEX_NAME,
            namespace="default",
            dimension=1536,
            metric="cosine",
  	        spec={
                  "serverless": {"region": "us-east-1", "cloud": "aws"}
            }
        )
        # Initialize Pinecone retriever through Haystack
        self.retriever = PineconeEmbeddingRetriever(
            document_store=self.document_store,
            top_k=5
        )
        
        # Define the RAG prompt template
        template = """
        Jesteś asystentem AI dla Skautów Europy, pomagającym w zadaniach administracyjnych. Odpowiadaj tylko w języku polskim.
        
        Proszę przestrzegaj tych wytycznych:
        - Odpowiadaj tylko na pytania związane z działalnością Skautów Europy
        - Bądź rzeczowy w swoich odpowiedziach, odpowiadaj wyczerpująco
        - Gdy nie masz pewności lub gdy nie znajdziesz w bazie powiązanych dokumentów kontekstowych - przyznaj, że nie masz wystarczających informacji
        - Opieraj swoje odpowiedzi na dostarczonym kontekście, nie na ogólnej wiedzy
        - Zawsze cytuj dokumenty zawierające istotne dla odpowiedzi informacje kontekstowe
        - Nie halucynuj. Nawiązuj wyłącznie do informacji z dokumentów kontekstowych.
        
        Ponizej zamieszczono pełny kontekst pytań, jakie będziesz otrzymywał:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}
        
        Pytanie użytkownika: {{ question }}
        
        Odpowiedź:
        """
        
        # Initialize prompt builder with the template
        self.prompt_builder = PromptBuilder(template=template, required_variables=[])
        
        # Initialize the LLM component with OpenAI
        self.generator = OpenAIGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model="gpt-4o-mini",
            generation_kwargs={"temperature": 0.2}
        )
        
        # Create the RAG pipeline
        self.rag_pipeline = Pipeline()
        self.rag_pipeline.add_component("text_embedder", OpenAITextEmbedder(api_key=Secret.from_token(OPENAI_API_KEY)))
        self.rag_pipeline.add_component("retriever", self.retriever)
        self.rag_pipeline.add_component("prompt_builder", self.prompt_builder)
        self.rag_pipeline.add_component("llm", self.generator)
        
        # Connect the components
        self.rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.rag_pipeline.connect("retriever", "prompt_builder.documents")
        self.rag_pipeline.connect("prompt_builder", "llm")
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Process a user question and return an answer using the RAG pipeline.
        
        Args:
            question: The user's question about Scouts d'Europe administration
            
        Returns:
            A dictionary containing the answer and metadata
        """
        
        # Run the RAG pipeline
        try:
            result = self.rag_pipeline.run({
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question}
            })
            
            # Extract retrieved documents for debugging/transparency
            retrieved_docs = []
            if hasattr(result, "retriever"):
                retrieved_docs = [
                    {"content": doc.content, "score": doc.score if hasattr(doc, "score") else None}
                    for doc in result["retriever"]
                ]
            
            return {
                "answer": result["llm"],
                "is_relevant": True,
                "retrieved_docs": retrieved_docs
            }
        except Exception as e:
            print("Error when processing question: \n")
            print(e)
            # Handle any pipeline errors
            return {
                "answer": f"I encountered an issue while processing your question. Please try rephrasing or ask another question about Scouts d'Europe administration.",
                "error": str(e),
                "is_relevant": True,
                "retrieved_docs": []
            }

# Command-line interface for testing
def main():
    """Simple CLI to test the Scouts Assistant."""
    assistant = ScoutsAssistant()
    
    print("Scouts d'Europe Administrative Assistant")
    print("----------------------------------------")
    print("Ask questions about scout administration, or type 'exit' to quit.")

    def handle_sigint(signum, frame):
        print("\nInterrupt received, exiting...")
        sys.exit(0)

    # Register the signal handler
    signal.signal(signal.SIGINT, handle_sigint)

    while True:
        user_input = input("\nQuestion: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        response = assistant.answer_question(user_input)
        print("\nAnswer:", response["answer"]['replies'][0])
        
        # Optionally show retrieved documents for debugging
        if response.get("retrieved_docs") and len(response["retrieved_docs"]) > 0:
            print("\nRetrieved sources:")
            for i, doc in enumerate(response["retrieved_docs"], 1):
                print(f"{i}. {doc['content'][:100]}... (score: {doc['score']})")


if __name__ == "__main__":
    main()