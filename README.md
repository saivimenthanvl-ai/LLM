# LLM
![image](https://github.com/user-attachments/assets/88f1afef-ecce-4ade-b32d-fc83eb9ab42d)
Inputs: The source data.
Input Embeddings: Each input token is transformed into a high-dimensional vector.
Positional Encoding: Since transformers lack inherent sequence order awareness, positional encodings are added to each embedding to let the model know where each token sits in the sequence.
Masked Multi-Head Attention Block
This is the core component for capturing relationships between different positions in the input sequence.
a. Linear Projections to Q, K, V
Each input vector (X) is linearly projected to form three vectors:
Q (Query)  Wq
K (Key) Wk
V (Value)  Wv
These are done for all tokens, yielding matrices Q, K, and V.
Attention Calculation
Dot Product: Compute dot products between Q and K for each possible pair of tokens. Measures how much focus each token should pay to the others.
Scale: Divide by the square root of the dimension of K to control the magnitude.
Mask: Apply a mask so each position cannot "see" future tokens
Softmax: Transforms the scaled scores into attention weights.
Weighted Sum: Multiply these weights with V to produce the attention outputs.
Multi-Head Mechanism
Several attention "heads" operate in parallel, each with its own Wq, Wk, Wv.
Each head learns to focus on different aspects/positions of the sequence.
Concatenation and Linear Layer
Outputs from all heads are concatenated.
This is passed through another linear transformation (Wo) to mix the information from all heads.
After Multi-Head Attention: Transformer Block
The Multi-Head Attention block is part of a larger Transformer decoder layer (the gray block on the left).
Add and Norm
The multi-head attention output is added to the original input (skip connection).
Layer normalization stabilizes training.
Feed Forward
The output goes through a position-wise feed-forward neural network.
Second Add and Norm
Another skip connection and normalization are applied.
Stacked Layers
Several identical layers form the full transformer decoder.
Output
The output of the last layer is passed through a linear layer and softmax to obtain the probability distribution for the next token.

