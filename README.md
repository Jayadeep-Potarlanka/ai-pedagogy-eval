# Assessing Pedagogical Capabilities of Large Language Models

This project assesses the effectiveness of various Large Language Models (LLMs) in performing pedagogical tasks. It uses transformer-based classifiers (DistilBERT and RoBERTa) to evaluate the quality of AI-generated tutor responses based on a dataset of student-tutor interactions.

## Project Overview

This project focuses on classifying AI tutor responses based on four key pedagogical dimensions. The goal is to evaluate how well different language models can identify mistakes, locate them, provide guidance, and offer actionable feedback.

### Pedagogical Tasks

The evaluation is based on four classification tasks that measure the quality of a tutor's response:
1.  **Mistake Identification**: Does the response correctly identify the student's error?
2.  **Mistake Location**: Does the response pinpoint where the error occurred in the student's solution?
3.  **Providing Guidance**: Does the response offer constructive guidance to help the student learn?
4.  **Actionability**: Is the guidance provided clear and actionable for the student?

## Models Explored and Dataset

### AI Tutors (LLMs) Assessed
The dataset contains responses from the following AI tutors, which are evaluated in this project:
*   Sonnet
*   Llama318B
*   Llama31405B
*   GPT4
*   Mistral
*   Gemini
*   Phi3
*   **Baselines**: Expert and Novice human tutors

### Classification Models
Three different modeling approaches were implemented and evaluated:

1.  **DistilBERT (`DistilBERT.ipynb`)**: A single DistilBERT-based model trained on the entire dataset to handle all four tasks for all tutors. This model uses Focal Loss to address class imbalance.
2.  **RoBERTa (`RoBERTa.ipynb`)**: A single RoBERTa-based model, also trained on the entire dataset for all tasks and tutors, using Focal Loss.
3.  **RoBERTa - Multi-Head (`RoBERTa-multi-head.ipynb`)**: A multi-head approach where a separate RoBERTa-based model is trained for each individual AI tutor (e.g., Sonnet, GPT4, etc.). This allows the model to specialize in the response style of a specific tutor. This approach also utilizes Focal Loss.

### Dataset
The project uses the `ai_tutors_dataset.json` file. This dataset contains conversation histories between students and various AI tutors, along with the tutors' responses. Each response is annotated for the four pedagogical tasks mentioned above.

## Methodology

The project employs a multi-task learning framework where models are trained to predict a label for each of the four tasks simultaneously.

*   **Input**: The model takes the conversation history and the tutor's response as input.
*   **Architecture**: All models use a transformer encoder (DistilBERT or RoBERTa) followed by four separate classification heads, one for each pedagogical task.
*   **Loss Function**: Focal Loss is used to mitigate the effects of class imbalance present in the dataset annotations.
*   **Evaluation**: Models are evaluated on a held-out validation set. Performance is measured using Accuracy and Macro F1-score under two schemes:
    *   **Exact (3-Class)**: "No", "To some extent", "Yes".
    *   **Lenient (2-Class)**: "No" vs. "To some extent" and "Yes" combined.

## Results

### DistilBERT vs. RoBERTa Performance

The performance of DistilBERT and RoBERTa is comparable, with each model excelling in different areas. RoBERTa generally shows a slight advantage in accuracy, but DistilBERT's performance is competitive, making it a viable, more efficient alternative.

| Task                   | Metric     | DistilBERT | RoBERTa        |
| :--------------------- | :--------- | :--------- | :------------- |
| **Mistake Identification** | Accuracy   | 0.8770    | **0.8911**    |
|                        | Macro F1   | 0.6761    | **0.6988**    |
| **Mistake Location**       | Accuracy   | 0.7440    | **0.7560**    |
|                        | Macro F1   | **0.5488**  | 0.5453      |
| **Providing Guidance**     | Accuracy   | **0.6754**  | 0.6714      |
|                        | Macro F1   | **0.5894**  | 0.5845      |
| **Actionability**          | Accuracy   | 0.7218    | **0.7359**    |
|                        | Macro F1   | 0.6394    | **0.6447**    |

## Repository Files

*   `DistilBERT.ipynb`: Contains the code for training and evaluating the multi-head DistilBERT model on the full dataset.
*   `RoBERTa.ipynb`: Contains the code for training and evaluating the multi-head RoBERTa model on the full dataset.
*   `RoBERTa-multi-head.ipynb`: Implements the per-tutor evaluation, training a separate RoBERTa model for each AI tutor.
*   `ai_tutors_dataset.json`: The core dataset containing the annotated student-tutor conversations.

## How to Run

1.  Ensure you have a Python environment with PyTorch, Transformers, and scikit-learn installed.
2.  Place the `ai_tutors_dataset.json` file in the same directory as the notebooks.
3.  Open and run the cells in any of the Jupyter Notebooks (`.ipynb` files) to replicate the training and evaluation process. It is recommended to use a GPU-accelerated environment for faster training.

