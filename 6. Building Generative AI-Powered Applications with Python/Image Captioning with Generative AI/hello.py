import gradio as gr

def greet(name):
    return "Hello " + name + "!, You are doing great!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch(server_name="0.0.0.0", server_port= 7860)
