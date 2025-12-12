'''
2025-12-10
Author: Dan Schumacher
How to run:
   python ./src/llamaEmbed.py 
'''
import numpy as np
import torch  # noqa: F401
from utils.llamaPrompter import LlamaVisionalPrompter, VisPrompt

def main():
    prompter = LlamaVisionalPrompter()
    vp1 = VisPrompt(
        user_text="Describe what is in this image",
        image_path="demo/images/ex1.jpg",
    )
    vp2 = VisPrompt(
        image_path="demo/images/ex1.jpg",
    )
    system_prompt = "You are a helpful assistant that accurately describes images."

    w_system_w_user = prompter.get_completion([vp1], system_prompt=system_prompt)
    w_user = prompter.get_completion([vp1])
    vis_only= prompter.get_completion([vp2])

    print("WITH SYSTEM AND USER PROMPT:\n", w_system_w_user)
    print("\nWITH USER PROMPT ONLY:\n", w_user)
    print("\nWITH VISUAL PROMPT ONLY:\n", vis_only)

    # embeddings = prompter.get_embedding([vp])
    # print("MODEL EMBEDS:", embeddings)
    # print(type(embeddings))
    # emb_np = embeddings.detach().to(torch.float32).cpu().numpy()

    # np.save("llamaEmbeddingTest.npy", emb_np)
    print("SAVED!")

if __name__ == "__main__":
    main()