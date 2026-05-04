import gradio as gr

def add_numbers(Num1, Num2):
    return Num1 + Num2

# Define the interface
demo = gr.Interface(
    fn=add_numbers, 
    inputs=["number", "number"], # Create two numerical input fields where users can enter numbers
    outputs="number" # Create numerical output fields
)

# Launch the interface
demo.launch(server_name="0.0.0.0", server_port= 7860)
