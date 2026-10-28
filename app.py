import gradio as gr
from citefaith.checker import CitationChecker

checker = CitationChecker()

def check_faithfulness(text, sources_text):
    if not text.strip():
        return "⚠️ Please enter text to verify."
    sources = [s.strip() for s in sources_text.split('\n') if s.strip()]
    if not sources:
        return "⚠️ Please provide at least one source."
    
    result = checker.check(text, sources)
    
    verdict_emoji = {'supported': '✅', 'contradicted': '❌', 'neutral': '⚪'}
    
    claims_md = []
    for c in result['claims']:
        emoji = verdict_emoji.get(c['verdict'], '❓')
        evidence = c['evidence'][:300] + ('...' if len(c['evidence']) > 300 else '')
        claims_md.append(
            f"{emoji} **{c['verdict'].upper()}** ({c['score']:.2f})\n"
            f"> {c['claim']}\n"
            f"> Evidence: {evidence}"
        )
    
    faithful = "✅ **FAITHFUL**" if result['faithful'] else "❌ **NOT FAITHFUL**"
    summary = f"{faithful}  \nClaims: {result['total_claims']} | Unsupported: {result['unsupported_count']}"
    
    return f"{summary}\n\n" + "\n\n".join(claims_md)


demo = gr.Interface(
    fn=check_faithfulness,
    inputs=[
        gr.Textbox(
            label="Text to verify",
            lines=6,
            placeholder="Paste the text you want to verify..."
        ),
        gr.Textbox(
            label="Sources (one per line)",
            lines=5,
            placeholder="Source 1\nSource 2\nSource 3..."
        )
    ],
    outputs=gr.Markdown(label="Results"),
    title="🔍 CiteFaith: Citation Faithfulness Checker",
    description=(
        "Verify claims against sources using NLI + semantic retrieval. "
        "Powered by DeBERTa-v3 NLI + MiniLM embeddings. "
        "Open-source models & data (RAGBench, HoVer, SciFact)."
    ),
    examples=[
        [
            "Marie Curie died in 1934 from aplastic anemia caused by radiation exposure.",
            "Marie Curie died in 1934 of aplastic anemia, likely caused by long exposure to radiation."
        ],
        [
            "The Eiffel Tower is located in London.",
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
        ]
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()