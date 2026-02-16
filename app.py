import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

MODEL_NAME = "google/mt5-small"
print("Loading pretrained model...\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Model loaded successfully ✔\n")

SUPPORTED_LANGUAGES = {
    "Hindi": sanscript.DEVANAGARI,
    "Telugu": sanscript.TELUGU,
    "Tamil": sanscript.TAMIL
}

def neural_demo(text, lang):
    prompt = f"Transliterate to {lang}: {text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=50, num_beams=4)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
def rule_based_transliteration(text, script):
    return transliterate(text, sanscript.ITRANS, script)

def transliterate_all(text):
    if not text.strip():
        
        return "Please enter text", "Please enter text", "Please enter text"
    outputs = []
    for lang, script in SUPPORTED_LANGUAGES.items():
        _ = neural_demo(text, lang)
        final_output = rule_based_transliteration(text, script)
        outputs.append(final_output)

    return outputs[0], outputs[1], outputs[2]


def clear_all():
    
    return "", "", "", ""


custom_css = """
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
}
.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #dcdcdc;
    margin-bottom: 30px;
}
.output-box textarea {
    font-size: 18px !important;
    font-weight: bold;
    text-align: center;
}
"""


with gr.Blocks(
    title="AI Transliteration System",
    theme=gr.themes.Base(primary_hue="blue"),
    css=custom_css
) as app:

    gr.HTML("<div class='main-title'>AI Transliteration System</div>")
    gr.HTML("<div class='subtitle'>Roman ➜ Hindi | Telugu | Tamil</div>")

    with gr.Group():
        text_input = gr.Textbox(
            label="Enter Roman Text",
            placeholder="Example: mera naam utkarsh hai",
            lines=2
        )

        with gr.Row():
            transliterate_btn = gr.Button("Transliterate", variant="primary")
            clear_btn = gr.Button("Clear")

    gr.Markdown("## Transliteration Output")

    with gr.Row():

        with gr.Column():
            gr.Markdown("### Hindi")
            hindi_output = gr.Textbox(
                lines=2,
                interactive=False,
                elem_classes="output-box"
            )

        with gr.Column():
            gr.Markdown("### Telugu")
            Telugu_output = gr.Textbox(
                lines=2,
                interactive=False,
                elem_classes="output-box"
            )

        with gr.Column():
            gr.Markdown("### Tamil")
            tamil_output = gr.Textbox(
                lines=2,
                interactive=False,
                elem_classes="output-box"
            )

    transliterate_btn.click(
        transliterate_all,
        inputs=text_input,
        outputs=[hindi_output, Telugu_output, tamil_output]
    )

    clear_btn.click(
        clear_all,
        outputs=[text_input, hindi_output, Telugu_output, tamil_output]
    )

    with gr.Tab("Examples"):
        gr.Examples(
            examples=[
                ["mera naam utkarsh hai"],
                ["bharat ek mahan desh hai"],
                ['na paru sasidhar reddy'],
                ["vanakkam eppadi irukeenga"],
            ],
            inputs=text_input,
        )
    with gr.Tab("ℹ About Project"):
        gr.Markdown(
            """
            ### Hybrid Multilingual Transliteration System
            🔹 Pretrained Seq2Seq Transformer (mT5)  
            🔹 Rule-based Indic Transliteration Engine  
            🔹 Supports Hindi, Telugu, Tamil  
            🔹 Built using Python & Gradio  
            ---
            Designed by Sasidhar.
            """
        )
if __name__ == "__main__":
    app.launch()
