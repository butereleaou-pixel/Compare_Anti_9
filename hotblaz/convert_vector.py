# Install required library if not installed
# pip install transformers torch

from transformers import AutoTokenizer, AutoModel
import torch

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"

# 1. Choose a model (e.g., GPT-2, BERT, or any transformer with embeddings)
model_name = "gpt2"  # You can replace with "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

#model = AutoModel.from_pretrained(model_name)

'''
def convert_token(text, device="cuda"):
    """
    Convert text into token -> embedding dict safely.
    Uses GPU for computation, avoids unsafe custom extensions.
    Returns embeddings as NumPy arrays on CPU.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]  # shape: [seq_len]

    # Get token embeddings on GPU
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

    # Ensure contiguous and float32
    token_embeddings = token_embeddings.contiguous().to(torch.float32)

    # Move to CPU safely
    emb_cpu = token_embeddings.cpu().numpy()  # shape: [seq_len, hidden_dim]

    # Convert ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Build token -> embedding dict
    token_vector_dict = {token: emb_cpu[i] for i, token in enumerate(tokens)}

    return token_vector_dict
'''

def convert_token(text, max_len=1024):
    """
    GPU 输出：
    - token_ids: [seq_len]
    - token_emb: [seq_len, 768]
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_len,   # 限制最大token
        truncation=True       # 超过自动截断
    ).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        token_emb = outputs.last_hidden_state[0]  # [seq_len, hidden]

    token_ids = inputs["input_ids"][0]           # [seq_len]

    return token_ids, token_emb    # 全在 GPU

'''

def convert_token(text):
    
    #Convert input text into token-vector pairs (dict format).
    #Uses a compiled C++/CUDA helper to pack/copy embeddings on GPU and return a single CPU tensor,
    #avoiding slow Python-level loops over tokens and per-vector tolist() calls.
   
    # Tokenize the text
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f"we are using {device} to convert")
    device = "cuda"
    inputs = tokenizer(text, return_tensors="pt").to(device)  # device = "cuda" ideally
    input_ids = inputs["input_ids"]

    # Get token embeddings (on GPU)
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

    # If already CPU-only and small, you can skip the extension:
    if not token_embeddings.is_cuda:
        # Fast path: single move to CPU and numpy
        emb_cpu = token_embeddings.contiguous().cpu().numpy()
    else:
        # Use the compiled extension: returns a CPU contiguous tensor
        import embedding_ext
        emb_cpu_tensor = embedding_ext.embedding_copy(token_embeddings)  # torch tensor on CPU
        emb_cpu = emb_cpu_tensor.numpy()  # single zero-copy view since contiguous on CPU

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Build dict using numpy slicing (fast) — this still creates Python objects for keys,
    # but avoids per-token deep conversion loops
    token_vector_dict = {}
    for i, token in enumerate(tokens):
        # If you prefer lists: emb_cpu[i].tolist()  <-- converting each row to a list is still O(n)
        token_vector_dict[token] = emb_cpu[i]  # NumPy 1D array view

    return token_vector_dict
'''

'''
def convert_token(text):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Get token embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state[0]  # Shape: [seq_len, hidden_dim]

    # Convert tokens and embeddings into a dictionary
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_vector_dict = {}

    for token, vector in zip(tokens, token_embeddings):
        # Convert PyTorch tensor to Python list
        token_vector_dict[token] = vector.tolist()  # e.g., [0.1, 0.2, ...]

    return token_vector_dict  # Returns a dict, not a list!
'''

'''
    with open(output_file, "w", encoding="utf-8") as f:
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        vectors = token_embeddings[0]

        for token, vector in zip(tokens, vectors):
            vector_str = ", ".join([str(v.item()) for v in vector])
            f.write(f"{token}\t{vector_str}\n")

    print(f"Saved token vectors to {output_file}")
'''

'''
text = "hello world !"

token = convert_token(text)

print ("convert token:", token)
'''
