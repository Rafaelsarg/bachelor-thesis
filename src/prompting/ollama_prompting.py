import ollama

class OllamaPrompting:
    """
    A wrapper for constructing and sending structured prompts to Ollama models.
    Supports zero-shot, few-shot, and custom prompting formats.
    """

    def __init__(self, model_name: str, system_instruction: str):
        """
        Initializes the OllamaPrompting instance with a model and base system prompt.

        Args:
            model_name (str): The Ollama model identifier (e.g., 'mistral:7b').
            system_instruction (str): The default instruction given to the model (system role).
        """
        self.model_name = model_name
        self.system_instruction = system_instruction
    
    def _create_base_system_message(self) -> list:
        """
        Creates the base message containing the system-level instruction.

        Returns:
            list: A list with one dictionary representing the system message.
        """
        return [{"role": "system", "content": self.system_instruction}]
    
    def send_prompt_to_model(self, message_sequence: list, temperature: float = 0.0) -> str:
        """
        Sends a list of structured messages to the Ollama model for response generation.

        Args:
            message_sequence (list): A list of messages with "role" and "content".
            temperature (float): Sampling temperature (0 = deterministic, 1 = creative).

        Returns:
            str: The content of the model's response.
        """
        response = ollama.chat(
            model=self.model_name,
            messages=message_sequence,
            options={"temperature": temperature}
        )
        return response["message"]["content"]
    
    def build_zero_shot_prompt(self, user_input_text: str) -> list:
        """
        Constructs a zero-shot prompt using the system message and a single user message.

        Args:
            user_input_text (str): The input text/question for the model.

        Returns:
            list: Structured messages for zero-shot inference.
        """
        messages = self._create_base_system_message()
        messages.append({"role": "user", "content": user_input_text})
        return messages
    
    def build_few_shot_prompt(self, user_input_text: str, example_pairs: list) -> list:
        """
        Constructs a few-shot prompt by appending labeled examples before the user query.

        Args:
            user_input_text (str): The current input/query to classify or respond to.
            example_pairs (list): A list of example dicts, each with 'input' and 'output' keys.

        Returns:
            list: Structured message list including examples and new query.

        Raises:
            ValueError: If the list of examples is empty.
        """
        if not example_pairs:
            raise ValueError("Few-shot prompting requires at least one example.")

        messages = self._create_base_system_message()

        # Insert prior examples as alternating user/assistant messages
        for example in example_pairs:
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})

        # Append the actual user query
        messages.append({"role": "user", "content": user_input_text})
        return messages
    
    def build_custom_prompt(self, custom_system_instruction: str, user_input_text: str, context_example: dict) -> list:
        """
        Constructs a custom prompt with a different system instruction and prior example.

        Args:
            custom_system_instruction (str): A custom system-level instruction to guide the model.
            user_input_text (str): The input/question to be processed.
            context_example (dict): A dictionary representing an example with a 'text' key.

        Returns:
            list: Structured message list for custom prompting.
        """
        messages = [{"role": "system", "content": custom_system_instruction}]
        messages.append({"role": "user", "content": context_example["text"]})
        messages.append({"role": "user", "content": user_input_text})
        return messages
