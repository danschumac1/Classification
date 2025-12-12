'''
CUDA_VISIBLE_DEVICES=2 python ./demo/testing.py
'''
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import dotenv
from PIL import Image
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration


@dataclass
class VisPrompt:
    image_path: Optional[str] = None
    user_text: Optional[str] = None
    assistant_text: Optional[str] = None
    image: Image.Image = field(init=False)
    messages: List[Dict] = field(init=False)

    def __post_init__(self):
        assert self.image_path or self.user_text, "At least one of image_path or user_text must be provided."

        # Load image once
        content = []
        if self.image_path:
            self.image = Image.open(self.image_path).convert("RGB")
            content.append({"type": "image"})
        # This VisPrompt contributes a *mini-conversation*:
        if self.user_text:
            content.append({"type": "text", "text": self.user_text})

        self.messages = [{"role": "user","content": content,}]

        if self.assistant_text:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": self.assistant_text,
                }
            )


class LlamaVisionPrompter:
    def __init__(self, system_prompt: str ='', model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", max_new_tokens=2056):
        dotenv.load_dotenv("./resources/.env", override=True)  # Load environment variables from .env file
        self.system_prompt = system_prompt
        self.model_id = model_id
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,  # transformers warns torch_dtype is deprecated
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_new_tokens = max_new_tokens  
        self.temperature = 0.1

    def create_inputs(
        self,
        vis_prompts: List[VisPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ) -> tuple[list[dict], list[Image.Image]]:
        """Build a flat chat list + aligned image list."""
        if not suppress_warnings:
            if not system_prompt:
                warnings.warn("No system prompt provided; model behavior may be unpredictable.", UserWarning)
        chat: List[Dict] = []
        images: List[Image.Image] = []

        # System message
        if system_prompt:
            chat.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        # Few-shot examples + query
        for vp in vis_prompts:
            # Flatten this VisPrompt's messages into the main chat
            chat.extend(vp.messages)
            # Each VisPrompt has exactly one user image
            images.append(vp.image)

        input_text = self.processor.apply_chat_template(
                chat,
                add_generation_prompt=True,
            )
        return input_text, images
    
    def create_inputs_batch(
        self,
        vis_prompts: List[VisPrompt],
        system_prompt: Optional[str] = None,
        suppress_warnings: bool = False,
    ) -> tuple[List[str], List[Image.Image]]:
        """
        Build one chat per VisPrompt (for true batching).
        Returns:
            input_texts: list[str]  -- one serialized chat per example
            images:      list[Image.Image] -- one image per example (same order)
        """
        if not suppress_warnings and not system_prompt:
            warnings.warn("No system prompt provided; model behavior may be unpredictable.", UserWarning)

        input_texts: List[str] = []
        images: List[Image.Image] = []

        for vp in vis_prompts:
            chat: List[Dict] = []
            if system_prompt:
                chat.append(
                    {
                        "role": "system",
                        "content": system_prompt,
                    }
                )
            chat.extend(vp.messages)

            text = self.processor.apply_chat_template(
                chat,
                add_generation_prompt=True,
            )
            input_texts.append(text)
            images.append(vp.image)

        return input_texts, images

    # -------------------------------
    # Helper: strip prompt + "assistant" header
    # -------------------------------
    def _postprocess_outputs(
        self,
        full_texts: List[str],
        prompts: List[str],
    ) -> List[str]:
        answers: List[str] = []
        for full, prompt in zip(full_texts, prompts):
            # Drop the prompt prefix if it appears verbatim
            if full.startswith(prompt):
                ans = full[len(prompt):].strip()
            else:
                ans = full.strip()

            # Strip leading "assistant" header if present
            lower = ans.lower()
            if lower.startswith("assistant"):
                # remove first occurrence of "assistant" token/word
                parts = ans.split("assistant", 1)
                ans = parts[1].strip() if len(parts) > 1 else ans

            answers.append(ans)
        return answers


    # -------------------------------
    # UPDATED: can do single or batch
    # -------------------------------
    def _strip_assistant_header(self, text: str) -> str:
        """
        Remove a leading 'assistant' header if the model starts its reply with it.
        """
        stripped = text.lstrip()
        lower = stripped.lower()
        if lower.startswith("assistant"):
            # Cut off the first line or the word
            parts = stripped.split("\n", 1)
            if len(parts) == 2:
                return parts[1].strip()
            else:
                return ""  # it was just "assistant"
        return text.strip()

    def get_completion(
        self,
        vis_prompts: List[VisPrompt],
        batch: bool = False,
    ):
        if batch:
            # True batching: one chat per example
            input_texts, images = self.create_inputs_batch(
                vis_prompts,
                system_prompt=self.system_prompt,
            )

            # Nested images: one list per example
            images_nested = [[img] for img in images]

            inputs = self.processor(
                images=images_nested,
                text=input_texts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )


            # Use attention_mask to get prompt lengths per example
            attn_mask = inputs["attention_mask"]  # (B, L)
            prompt_lens = attn_mask.sum(dim=1)    # (B,)

            answers: list[str] = []
            for i in range(output_ids.size(0)):
                plen = int(prompt_lens[i].item())
                gen_ids = output_ids[i, plen:]  # only new tokens
                text = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
                text = self._strip_assistant_header(text)
                answers.append(text)

            return answers

        else:
            # Original: one conversation (can include few-shot)
            input_text, images = self.create_inputs(
                vis_prompts,
                system_prompt=self.system_prompt,
            )

            # Single conversation may have multiple images → flat list is fine here
            inputs = self.processor(
                images=images,
                text=input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
            # Here prompt_len is shared for the single batch element
            attn_mask = inputs["attention_mask"]  # (1, L)
            prompt_len = int(attn_mask.sum(dim=1)[0].item())
            gen_ids = output_ids[0, prompt_len:]
            text = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
            text = self._strip_assistant_header(text)
            return text

    def get_embedding(
        self,
        vis_prompts: List[VisPrompt],
        system_prompt: Optional[str] = None,
        batch: bool = False,
        layer: int = -1,                    # which hidden layer to extract
    ) -> torch.Tensor:
        """
        Compute embeddings for one or more VisPrompts.
        
        If batch=False (default):
            - Behaves like old get_embedding (few-shot, one conversation).
            - Returns a tensor of shape (hidden_dim,).

        If batch=True:
            - Treats each VisPrompt independently (true batching).
            - Returns a tensor of shape (batch_size, hidden_dim).
        """

        # -----------------------------
        # BATCH MODE
        # -----------------------------
        if batch:
            # Build one chat per VisPrompt
            input_texts, images = self.create_inputs_batch(
                vis_prompts,
                system_prompt=system_prompt or self.system_prompt,
            )

            # Processor will automatically batch text + images
            inputs = self.processor(
                images=images,
                text=input_texts,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                )
                # outputs.hidden_states[layer] -> (batch, seq_len, hidden_dim)
                hidden = outputs.hidden_states[layer]
                # Use [CLS] token → index 0
                embeddings = hidden[:, 0, :]   # (batch, hidden_dim)

            return embeddings  # (B, D)

        # -----------------------------
        # SINGLE MODE (few-shot)
        # -----------------------------
        else:
            input_text, images = self.create_inputs(
                vis_prompts,
                system_prompt=system_prompt or self.system_prompt,
            )

            inputs = self.processor(
                images=images,
                text=input_text,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[layer]
                embedding = hidden[:, 0, :]      # shape: (1, D)
                embedding = embedding.squeeze(0) # → (D,)

            return embedding  # (D,)