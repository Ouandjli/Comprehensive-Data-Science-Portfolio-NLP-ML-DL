import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)

# Prompts
prompt1 = "To be, or not to be: that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles, And by opposing end them?"
prompt2 = "All the world's a stage, and all the men and women merely players. They have their exits and their entrances; And one man in his time plays many parts."

# Tokenize the prompts
inputs1 = tokenizer(prompt1, return_tensors="pt")
inputs2 = tokenizer(prompt2, return_tensors="pt")

# Pass inputs through the model to get the attention weights
outputs1 = model(**inputs1)
outputs2 = model(**inputs2)

# Extract attention weights
attentions1 = outputs1.attentions
attentions2 = outputs2.attentions


# Function to plot attention weights
def plot_attention(attentions, title):
    # Take the attention from the last layer and average over heads
    attention = attentions[-1].mean(dim=1).squeeze().detach().numpy()

    # Convert token IDs to tokens for display on axes
    tokens = tokenizer.convert_ids_to_tokens(inputs1.input_ids[0])

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(attention, cmap='viridis')

    # Set ticks for both x and y axes
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))

    # Label ticks with tokens, rotate x labels for readability
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)

    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.title(title)
    fig.colorbar(cax)
    plt.show()


# Plot attention matrices for both prompts
plot_attention(attentions1, "Attention Weights for Prompt 1")
plot_attention(attentions2, "Attention Weights for Prompt 2")