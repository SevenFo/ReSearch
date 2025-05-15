import requests
import json
import logging
from typing import List, Dict, Any
from flashrag.retriever.retriever import retry

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Re-using the prompt template string
DEFAULT_RELEVANCY_PROMPT_TEMPLATE_STR = """As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.

Retrieved Document:
-------------------
{context_str}

User Question:
--------------
{query_str}

Evaluation Criteria:
- Consider whether the document contains keywords or topics related to the user's question.
- The evaluation should not be overly stringent; the primary objective is to identify and filter out clearly irrelevant retrievals.

Decision:
- Assign a binary score to indicate the document's relevance.
- Use 'yes' if the document is relevant to the question, or 'no' if it is not.

Please provide your binary score ('yes' or 'no') below to indicate the document's relevance to the user question."""


class OllamaDirectRAGEvaluator:
    def __init__(
        self,
        ollama_model_name: str,
        ollama_base_url: str = "http://localhost:11434/v1",  # Default Ollama OpenAI-compatible endpoint
        temperature: float = 0.0,  # For deterministic output
        timeout: int = 120,  # seconds
        relevancy_prompt_template: str = DEFAULT_RELEVANCY_PROMPT_TEMPLATE_STR,
    ):
        """
        Initializes an evaluator that uses an Ollama-served LLM (via direct OpenAI API calls)
        to assess document relevance.

        Args:
            ollama_model_name (str): The name of the model served by Ollama (e.g., "llama3", "mistral").
            ollama_base_url (str): The base URL for the Ollama OpenAI-compatible API.
            temperature (float): Temperature for the LLM. 0.0 is good for classification.
            timeout (int): Request timeout in seconds.
            relevancy_prompt_template (str): The prompt template string to use for evaluation.
        """
        self.model_name = ollama_model_name
        self.base_url = ollama_base_url.rstrip("/")  # Ensure no trailing slash
        self.chat_completions_url = f"{self.base_url}/chat/completions"
        self.temperature = temperature
        self.timeout = timeout
        self.relevancy_prompt_template = relevancy_prompt_template
        logging.info(
            f"OllamaDirectRAGEvaluator initialized with model: {ollama_model_name} at {self.base_url}"
        )

    def _construct_prompt(self, query_str: str, context_str: str) -> str:
        """Constructs the full prompt for the LLM."""
        return self.relevancy_prompt_template.format(
            query_str=query_str, context_str=context_str
        )

    @retry(max=5, sleep=1)
    def _call_ollama_api(self, prompt_content: str) -> str:
        """
        Makes a POST request to the Ollama /v1/chat/completions endpoint.
        """
        headers = {
            "Content-Type": "application/json",
            # "Authorization": "Bearer ollama" # Ollama typically doesn't require auth for local API
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt_content}],
            "temperature": self.temperature,
            "max_tokens": 10,  # Limit response length
            "stream": False,  # We want a single response for this use case
        }

        try:
            response = requests.post(
                self.chat_completions_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            response_data = response.json()

            # Extract the message content
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                message = response_data["choices"][0].get("message", {})
                content = message.get("content", "")
                return content.strip().lower()
            else:
                logging.error(f"Unexpected API response structure: {response_data}")
                return ""
        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to decode JSON response: {e} - Response text: {response.text}"
            )
            return ""

    def evaluate_retrieval(self, query: str, docs: List[str]) -> List[str]:
        """
        Evaluates the relevance of a list of documents to a given query.

        Args:
            query (str): The user's query.
            docs (List[str]): A list of document texts to evaluate.

        Returns:
            List[str]: A list of strings, where each string is 'yes' or 'no',
                       indicating the relevance of the corresponding document.
        """
        relevancy_results = []
        for i, doc_text in enumerate(docs):
            if not doc_text or not doc_text.strip():
                logging.warning(
                    f"Document {i + 1} is empty or whitespace-only. Marking as 'no'."
                )
                relevancy_results.append("no")
                continue

            logging.debug(
                f"Evaluating relevance for doc {i + 1}/{len(docs)} against query: '{query}'"
            )
            full_prompt = self._construct_prompt(query_str=query, context_str=doc_text)
            llm_output = self._call_ollama_api(full_prompt)

            if "yes" in llm_output:
                relevancy_results.append(1.0)
            elif "no" in llm_output:
                relevancy_results.append(-1.0)
            else:
                logging.warning(
                    f"LLM returned an unexpected relevancy value: '{llm_output}'. "
                    f"Query: '{query}', Doc: '{doc_text[:100]}...'. Defaulting to 'no'."
                )
                relevancy_results.append(
                    -1.0
                )  # Default to 'no' if output is not 'yes' or 'no'

        print(f"Relevance evaluation for query '{query}': {relevancy_results}")
        return relevancy_results


# --- Example Usage ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # IMPORTANT: Make sure Ollama is running and the model is available.
    # OLLAMA_MODEL = "mistral"
    OLLAMA_MODEL = "qwen2:0.5b"  # Using a smaller model for quicker local testing

    print(f"Attempting to use Ollama model: {OLLAMA_MODEL}")
    print(
        "Ensure Ollama is running and the model is pulled (e.g., `ollama pull qwen2:0.5b`)."
    )
    print("Ollama OpenAI-compatible API should be at http://localhost:11434/v1")

    try:
        evaluator = OllamaDirectRAGEvaluator(ollama_model_name=OLLAMA_MODEL)

        query1 = "What is the main component of stars?"
        docs1 = [
            "Stars are primarily composed of hydrogen and helium.",
            "The sun is a star located at the center of our solar system.",
            "Photosynthesis is a process used by plants to convert light energy.",
            "Helium is the second lightest element in the periodic table.",
            "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
        ]
        relevance1 = evaluator.evaluate_retrieval(query1, docs1)
        print(f"\nQuery: {query1}")
        for doc, rel in zip(docs1, relevance1):
            print(f"  Relevance: {rel} - Doc: {doc[:70]}...")

        query2 = "Tell me about common house pets."
        docs2 = [
            "The history of the internet began with the development of electronic computers in the 1950s.",
            "Dogs and cats are popular household pets, known for their companionship.",
            "A balanced diet for a cat should include proteins, fats, and vitamins.",
            "Artificial intelligence is a branch of computer science.",
        ]
        relevance2 = evaluator.evaluate_retrieval(query2, docs2)
        print(f"\nQuery: {query2}")
        for doc, rel in zip(docs2, relevance2):
            print(f"  Relevance: {rel} - Doc: {doc[:70]}...")

        query3 = "Any topic"
        docs3 = ["This is a valid document.", "", "  ", "Another valid one."]
        relevance3 = evaluator.evaluate_retrieval(query3, docs3)
        print(f"\nQuery: {query3}")
        for doc, rel in zip(docs3, relevance3):
            print(f"  Relevance: {rel} - Doc: '{doc[:70]}...'")

    except Exception as e:
        print(f"\nAn error occurred during the example usage: {e}")
        print(
            "Please ensure Ollama is running, the model is downloaded, and network is accessible."
        )
        print(
            "You might need to run 'ollama pull qwen2:0.5b' (or your chosen model) first."
        )
