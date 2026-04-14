import gradio as gr
import skops.io as sio

untrusted = sio.get_untrusted_types(file="./Model/drug_pipeline.skops")
pipe = sio.load("./Model/drug_pipeline.skops", trusted=untrusted)

DRUG_INFO = {
    "DrugY": ("DrugY", "#00C9A7", "#005C4B", "Typically prescribed for patients with high Na_to_K ratio (>14)."),
    "drugA": ("Drug A", "#845EC2", "#3B1F6E", "Often used for older patients with HIGH blood pressure and HIGH cholesterol."),
    "drugB": ("Drug B", "#FF6F91", "#7A1A35", "Prescribed for older patients with HIGH blood pressure and NORMAL cholesterol."),
    "drugC": ("Drug C", "#FFC75F", "#7A5500", "Used for younger patients with LOW blood pressure."),
    "drugX": ("Drug X", "#0089BA", "#003F5A", "Prescribed for patients with NORMAL blood pressure and low Na_to_K ratio."),
}


def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    if sex is None or blood_pressure is None or cholesterol is None:
        return (
            gr.update(value="""
            <div style="text-align:center; padding:30px; color:#FF6F91; font-size:14px; font-weight:600;
                        background:rgba(255,111,145,0.08); border-radius:12px; border:1px dashed #FF6F91;">
                ⚠️ Please fill in all fields before predicting.
            </div>""", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
    predicted = pipe.predict([features])[0]
    name, color, dark, info = DRUG_INFO.get(predicted, (predicted, "#845EC2", "#3B1F6E", ""))

    result_html = f"""
    <div style="
        background: linear-gradient(135deg, {color}33 0%, {color}11 100%);
        border: 1.5px solid {color}66;
        border-radius: 16px;
        padding: 24px 28px;
        margin-top: 8px;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position:absolute; top:-20px; right:-20px;
            width:100px; height:100px;
            background: radial-gradient(circle, {color}44, transparent 70%);
            border-radius: 50%;
        "></div>
        <div style="font-size:11px; color:{color}; text-transform:uppercase; letter-spacing:2px; font-weight:700; margin-bottom:6px;">
            Recommended Drug
        </div>
        <div style="font-size:38px; font-weight:800; color:{color}; line-height:1.1; margin-bottom:10px;">
            {name}
        </div>
        <div style="
            display:inline-block; background:{color}22; color:{dark};
            border-radius:20px; padding:4px 12px; font-size:12px; font-weight:600;
            border: 1px solid {color}55;
        ">Clinical Note</div>
        <div style="font-size:13px; color:#555; margin-top:10px; line-height:1.6;">{info}</div>
    </div>
    """

    summary_html = f"""
    <div style="
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 16px 20px; margin-top: 12px;
        font-size: 13px;
    ">
        <div style="color:#aaa; font-size:11px; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px; font-weight:600;">
            Patient Summary
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:10px;">
            <span style="background:#ffffff15; color:#e0e0e0; padding:4px 12px; border-radius:20px;">Age: <b style="color:white">{age}</b></span>
            <span style="background:#ffffff15; color:#e0e0e0; padding:4px 12px; border-radius:20px;">Sex: <b style="color:white">{sex}</b></span>
            <span style="background:#ffffff15; color:#e0e0e0; padding:4px 12px; border-radius:20px;">BP: <b style="color:white">{blood_pressure}</b></span>
            <span style="background:#ffffff15; color:#e0e0e0; padding:4px 12px; border-radius:20px;">Cholesterol: <b style="color:white">{cholesterol}</b></span>
            <span style="background:#ffffff15; color:#e0e0e0; padding:4px 12px; border-radius:20px;">Na/K: <b style="color:white">{na_to_k_ratio}</b></span>
        </div>
    </div>
    """

    return (
        gr.update(value=result_html, visible=True),
        gr.update(value=summary_html, visible=True),
        gr.update(visible=True),
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

body, .gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}

.card {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 20px !important;
    padding: 28px !important;
    backdrop-filter: blur(12px) !important;
}

/* Slider track */
input[type=range] { accent-color: #845EC2; }

/* Radio buttons */
.gr-radio label { color: #ccc !important; }

/* Labels */
label span { color: #ccc !important; }

/* Buttons */
.lg.primary {
    background: linear-gradient(135deg, #845EC2, #FF6F91) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 20px rgba(132,94,194,0.4) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.lg.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(132,94,194,0.55) !important;
}

.secondary {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #ccc !important;
    border-radius: 20px !important;
    font-size: 12px !important;
}
.secondary:hover {
    background: rgba(255,255,255,0.13) !important;
    color: white !important;
}

footer { display: none !important; }

.divider { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 12px 0 24px; }
"""

with gr.Blocks(title="Drug Classification") as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding: 32px 0 16px;">
        <div style="
            display:inline-flex; align-items:center; justify-content:center;
            width:64px; height:64px; border-radius:18px; font-size:30px;
            background: linear-gradient(135deg, #845EC2, #FF6F91);
            box-shadow: 0 8px 24px rgba(132,94,194,0.5);
            margin-bottom:16px;
        ">💊</div>
        <h1 style="margin:0; font-size:28px; font-weight:800;
                   background: linear-gradient(90deg, #845EC2, #FF6F91, #FFC75F);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Drug Classification
        </h1>
        <p style="margin:10px 0 0; color:rgba(255,255,255,0.45); font-size:14px;">
            AI-powered drug recommendation based on patient vitals
        </p>
    </div>
    """)

    gr.HTML("<hr class='divider'>")

    with gr.Row(equal_height=True):

        # ── Left panel ───────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="card"):
            gr.HTML("<div style='font-size:16px; font-weight:700; color:white; margin-bottom:18px;'>Patient Information</div>")

            age = gr.Slider(
                minimum=15, maximum=74, step=1, value=35,
                label="Age", info="Range: 15 – 74 years"
            )

            with gr.Row():
                sex = gr.Radio(["M", "F"], label="Sex", value=None)
                cholesterol = gr.Radio(["HIGH", "NORMAL"], label="Cholesterol", value=None)

            blood_pressure = gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure", value=None)

            na_to_k = gr.Slider(
                minimum=6.2, maximum=38.2, step=0.1, value=15.0,
                label="Na / K Ratio", info="Sodium-to-potassium ratio in blood"
            )

            predict_btn = gr.Button("Predict Drug", variant="primary", size="lg")

            gr.HTML("<div style='font-size:12px; color:rgba(255,255,255,0.35); margin:16px 0 8px; text-transform:uppercase; letter-spacing:1px;'>Quick Examples</div>")
            with gr.Row():
                ex1 = gr.Button("High BP", size="sm", variant="secondary")
                ex2 = gr.Button("Low BP", size="sm", variant="secondary")
                ex3 = gr.Button("High Na/K", size="sm", variant="secondary")

        # ── Right panel ──────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="card"):
            gr.HTML("<div style='font-size:16px; font-weight:700; color:white; margin-bottom:18px;'>Prediction Result</div>")

            result_box = gr.HTML(
                value="""
                <div style="
                    text-align:center; padding:50px 20px;
                    color:rgba(255,255,255,0.2); font-size:13px;
                    border: 1px dashed rgba(255,255,255,0.1);
                    border-radius:14px;
                    background: rgba(255,255,255,0.02);
                ">
                    Fill in patient details and click<br>
                    <b style="color:rgba(255,255,255,0.35);">Predict Drug</b>
                </div>""",
            )
            summary_box = gr.HTML(visible=False)
            disclaimer = gr.HTML(
                value="""
                <div style="font-size:11px; color:rgba(255,255,255,0.3); margin-top:14px;
                            padding:10px 14px; background:rgba(255,255,255,0.03);
                            border-radius:8px; border:1px solid rgba(255,255,255,0.07);">
                    ⚠️ For educational purposes only. Always consult a licensed physician.
                </div>""",
                visible=False,
            )

    # ── Footer ───────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; margin-top:28px; font-size:11px; color:rgba(255,255,255,0.2);">
        Built with Gradio &nbsp;·&nbsp; CI/CD for Machine Learning &nbsp;·&nbsp; Powered by scikit-learn
    </div>
    """)

    # ── Handlers ─────────────────────────────────────────────────────────────
    predict_btn.click(
        fn=predict_drug,
        inputs=[age, sex, blood_pressure, cholesterol, na_to_k],
        outputs=[result_box, summary_box, disclaimer],
    )

    ex1.click(fn=lambda: (50, "M", "HIGH", "HIGH", 10.2),  outputs=[age, sex, blood_pressure, cholesterol, na_to_k])
    ex2.click(fn=lambda: (35, "F", "LOW", "NORMAL", 8.0),  outputs=[age, sex, blood_pressure, cholesterol, na_to_k])
    ex3.click(fn=lambda: (28, "M", "NORMAL", "HIGH", 30.5), outputs=[age, sex, blood_pressure, cholesterol, na_to_k])


demo.launch(
    theme=gr.themes.Base(
        primary_hue="purple",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
    css=CSS,
)
