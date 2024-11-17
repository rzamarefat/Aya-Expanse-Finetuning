from typing import Any
import gradio as gr
from AYA import AYA

class App:
    def __init__(self):
        self._translator_model = AYA()

        self._app = self._create_app()

    def _translate_text(self, content):
        german, persian = self._translator_model.infer(content)

        return german, persian

    def _create_app(self):
        with gr.Blocks() as app:
            gr.Markdown("### Translate Text to German and Persian")
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Write a paragraph or text here...",
                lines=4
            )
            translate_button = gr.Button("Translate")
            german_output = gr.Textbox(
                label="German Translation",
                lines=4
            )
            persian_output = gr.Textbox(
                label="Persian Translation",
                lines=4,
                rtl=True
            )
            translate_button.click(self._translate_text, inputs=[input_text], outputs=[german_output, persian_output])

        return app

    def __call__(self):
        self._app.launch()

