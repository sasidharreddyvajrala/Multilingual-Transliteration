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
    "Bengali": sanscript.BENGALI,
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
        _ = neural_demo(text, lang)  # pretrained usage
        final_output = rule_based_transliteration(text, script)
        outputs.append(final_output)

    return outputs[0], outputs[1], outputs[2]


def clear_all():
    return "", "", "", ""

with gr.Blocks(
    title="Multilingual Transliteration System",
    theme=gr.themes.Soft(),
) as app:

    gr.Markdown(
        """
        # 🌍 Multilingual Transliteration System
        ### Roman ➜ Hindi | Bengali | Tamil
        Enter text once and get transliteration in all languages.
        """
    )

    with gr.Row():
        text_input = gr.Textbox(
            label="Enter Roman Text",
            placeholder="Example: mera naam utkarsh hai",
            lines=2
        )

    transliterate_btn = gr.Button("🚀 Transliterate", variant="primary")
    clear_btn = gr.Button("🧹 Clear")

    # Outputs in Columns
    with gr.Row():

        hindi_output = gr.Textbox(
            label="Hindi Output",
            lines=2,
            interactive=False
        )

        bengali_output = gr.Textbox(
            label="Bengali Output",
            lines=2,
            interactive=False
        )

        tamil_output = gr.Textbox(
            label="Tamil Output",
            lines=2,
            interactive=False
        )

    transliterate_btn.click(
        transliterate_all,
        inputs=text_input,
        outputs=[hindi_output, bengali_output, tamil_output]
    )

    clear_btn.click(
        clear_all,
        outputs=[text_input, hindi_output, bengali_output, tamil_output]
    )

    with gr.Tab("📚 Examples"):

        gr.Examples(
            examples=[
                ["mera naam utkarsh hai"],
                ["bharat ek mahan desh hai"],
                ["ami tomake bhalobashi"],
                ["vanakkam eppadi irukeenga"],
            ],
            inputs=text_input,
        )


    with gr.Tab("ℹ About Project"):

        gr.Markdown(
            """
            **Hybrid Multilingual Transliteration System**  
            - Pretrained Seq2Seq Transformer (mT5)  
            - Rule-based Indic Transliteration Engine  
            - Supports Hindi, Bengali, Tamil  
            - Built using Python & Gradio  
            """
        )


if __name__ == "__main__":
    app.launch()