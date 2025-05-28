import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("neulab/python_bleu")
launch_gradio_widget(module)