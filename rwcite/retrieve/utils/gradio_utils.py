from transformers import StoppingCriteria
import sys


# Handle termination signal
def signal_handler(sig, frame):
    print("\nTermination signal received. Shutting down Gradio interface.")
    sys.exit(0)

# Custom stopping criteria
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        stop_ids = [29, 0]  # Define specific stop token IDs
        return input_ids[0][-1] in stop_ids

# Toggle task selection
def toggle_selection(current_task, new_task):
    """Toggle task selection: deselect if clicked again, otherwise update selection."""
    updated_task = "" if current_task == new_task else new_task
    return updated_task
