import streamlit as st
import sys
sys.path.insert(0, "src")

from citefaith.checker import CitationChecker

checker = CitationChecker()

st.set_page_config(page_title="CiteFaith", page_icon="🔍", layout="wide")
st.title("🔍 CiteFaith: Citation Faithfulness Checker")

st.markdown("Verify claims against sources using NLI + semantic retrieval.")

text = st.text_area("Text to verify", height=150, placeholder="Paste the text you want to verify...")
sources = st.text_area("Sources (one per line)", height=150, placeholder="Source 1\nSource 2\nSource 3...")

if st.button("Check Faithfulness", type="primary"):
    if not text.strip():
        st.warning("Please enter text to verify.")
    elif not sources.strip():
        st.warning("Please provide at least one source.")
    else:
        sources_list = [s.strip() for s in sources.split('\n') if s.strip()]
        if not sources_list:
            st.warning("Please provide at least one source.")
        else:
            with st.spinner("Checking..."):
                result = checker.check(text, sources_list)
            
            st.subheader("Results")
            verdict_emoji = {'supported': '✅', 'contradicted': '❌', 'neutral': '⚪'}
            
            for c in result['claims']:
                emoji = verdict_emoji.get(c['verdict'], '❓')
                st.markdown(f"{emoji} **{c['verdict'].upper()}** ({c['score']:.2f})")
                st.write(f"> {c['claim']}")
                evidence = c['evidence'][:300] + ('...' if len(c['evidence']) > 300 else '')
                st.write(f"> Evidence: {evidence}")
                st.divider()
            
            faithful = "✅ **FAITHFUL**" if result['faithful'] else "❌ **NOT FAITHFUL**"
            st.markdown(f"### {faithful}")
            st.metric("Total Claims", result['total_claims'])
            st.metric("Unsupported", result['unsupported_count'])

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Adjust thresholds in environment variables:")
    st.code("ENT_THRESHOLD=0.5\nCONTRA_THRESHOLD=0.5\nCONTRA_MIN_SIM=0.5\nSIM_THRESHOLD=0.7")
    st.divider()
    st.caption("Models:")
    st.code("NLI: DeBERTa-v3-small\nRetriever: MiniLM-L6-v2")