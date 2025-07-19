from transformers import pipeline
import torch # Import torch to check for GPU availability

def initialize_summarizer(model_name="sshleifer/distilbart-cnn-12-6"):
    """
    Initializes the Hugging Face summarization pipeline with a specified model.

    Args:
        model_name (str): The name of the pre-trained model to use for summarization.

    Returns:
        transformers.pipelines.text2text_generation.SummarizationPipeline:
            A summarization pipeline object.
    """
    # Determine the device to use (GPU if available, otherwise CPU)
    device = 0 if torch.cuda.is_available() else -1 # 0 for first GPU, -1 for CPU

    print(f"Loading summarization model: {model_name}...")
    try:
        # Pass the device argument to the pipeline
        summarizer = pipeline("summarization", model=model_name, device=device)
        print("Model loaded successfully!")
        if device == 0:
            print("Using GPU for summarization.")
        else:
            print("Using CPU for summarization.")
        return summarizer
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an active internet connection and the model name is correct.")
        print("Also, check your PyTorch/TensorFlow installation and GPU drivers if you intend to use a GPU.")
        return None

def summarize_paragraph(summarizer, text, min_length=30, max_length=150):
    """
    Summarizes a given text using the provided summarization pipeline.

    Args:
        summarizer (transformers.pipelines.text2text_generation.SummarizationPipeline):
            The initialized summarization pipeline.
        text (str): The input paragraph to summarize.
        min_length (int): The minimum length of the generated summary.
        max_length (int): The maximum length of the generated summary.

    Returns:
        str: The summarized text, or an error message if summarization fails.
    """
    if not summarizer:
        return "Summarizer not initialized. Cannot summarize."

    if not text.strip():
        return "Please provide some text to summarize."

    try:
        # Generate the summary.
        # do_sample=False makes the output deterministic for the same input.
        # You can set do_sample=True for more varied outputs.
        summary = summarizer(
            text,
            min_length=min_length,
            max_length=max_length,
            do_sample=False
        )
        # The output is a list of dictionaries, we want the 'summary_text' from the first item.
        return summary[0]['summary_text']
    except Exception as e:
        return f"An error occurred during summarization: {e}"