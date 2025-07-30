import json
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from google import genai
from google.genai import types
import os
import threading
import logging
from typing import List
import time


class LLMProcessor:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        """
        Initialize the LLMProcessor with a model, tokenizer, and system prompt.
        
        Args:
            model_name (str): Name of the model to load from Hugging Face.
            system_prompt (str): System prompt for processing legal texts.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.device
        ).eval()



    def process(self, user_input, system_prompt, history=None):
        """
        Process user input with the system prompt and conversation history.
        
        Args:
            user_input (str): The user's input text (e.g., legal document).
            history (list): List of tuples [(user_msg, assistant_msg), ...] for conversation history.
        
        Returns:
            str: The model's response or an error message if processing fails.
        """
        # Format the conversation history
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": user_input})

        # Apply chat template and tokenize
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.device)


        outputs = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=8192,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )

        # Decode and extract the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(input_text):].strip()


        return response

    def answer_question(
        self, question: str, history: List[List[str]] = None, top_k: int = 3, context: str = "", sys_prompt: str = ""
    ):
        """
        Answer a question with streaming output using TextIteratorStreamer.
        
        Args:
            question: User question
            history: Chat history as list of [user, assistant] pairs
            top_k: Number of recent history turns to include
            context: Optional context for the question
        
        Yields:
            Token chunks for streaming response
        """
        try:
            # Initialize messages with system prompt
            messages = [
                {
                    "role": "system",
                    "content": sys_prompt
                }
            ]

            # Add chat history if provided, limited to the last top_k turns
            if history:
                recent_history = history[-top_k:]
                for user_msg, assistant_msg in recent_history:
                    messages.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        messages.append({"role": "assistant", "content": assistant_msg})

            # Add current question with context
            # messages.append({"role": "user", "content": f"{question}"})
            messages.append({"role": "user", "content": f"### Context: {context}\n\n### Question: {question}"})

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize and move to the model's device
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # Initialize TextIteratorStreamer
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            # Generation parameters
            generation_kwargs = {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "max_new_tokens": 8192,
                "do_sample": True,
                "top_k": 10,
                "top_p": 0.95,
                "temperature": 0.4,
                "num_return_sequences": 1,
                "streamer": streamer,
            }

            # Run generation in a separate thread
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Yield tokens from the streamer
            for token in streamer:
                if token.strip():  # Yield only non-empty tokens
                    yield token

            # Wait for the generation thread to finish
            thread.join()

            # Clear CUDA cache after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            # logger.error(f"Error in answer_question: {str(e)}")
            yield f"Lỗi: {str(e)}"


class ApiProcessor:
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", api_key: str = None):
        """
        Initialize the ApiProcessor with a Gemini model and API key.

        Args:
            model_name (str): The name of the Gemini model to use.
                              Defaults to "gemini-2.5-flash-lite".
            api_key (str, optional): Your Google Gemini API key.
                                     If None, the processor will attempt to read
                                     the API key from the GEMINI_API_KEY
                                     environment variable.
        """
        if api_key:
            api_key = api_key
        else:
            # Attempt to read API key from the environment variable
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini API key not found. "
                    "Please provide it as an argument or set the GEMINI_API_KEY "
                    "environment variable."
                )
        self.client = genai.Client(api_key=api_key)



    def process(self, user_input: str, system_prompt: str = "") -> str:
        """
        Process user input with a system prompt and optional conversation history
        using the Gemini API's `generate_content` method.

        This method constructs the `contents` list required by the Gemini API,
        including previous turns from the `history` and the current `user_input`.
        The `system_prompt` is passed via `generation_config.system_instruction`.

        Args:
            user_input (str): The current input text from the user.
            system_prompt (str): The system instruction for the model, guiding its behavior.
                                 Defaults to an empty string.
            history (list, optional): A list of tuples, where each tuple represents
                                      a previous turn in the conversation:
                                      `[(user_message_1, assistant_response_1), ...]`.
                                      Defaults to None.

        Returns:
            str: The model's generated response text. Returns an error message
                 if the API call fails.
        """
        # The 'contents' parameter for generate_content expects a list of Content objects.
        # Each Content object represents a turn in the conversation with a role (user/model)
        try:
            # Add previous conversation turns from history
            response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
            )
            return response.text
        except Exception as e:
            # Handle any exceptions that occur during the API call
            return f"Error processing request with Gemini API: {e}"






            