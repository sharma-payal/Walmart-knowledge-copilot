"""
Walmart Associate Knowledge Copilot - Streamlit Interface
"""

import streamlit as st
import sys
import time
from datetime import datetime

sys.path.append('src')
from rag_pipeline import WalmartRAGPipeline

st.set_page_config(
    page_title="Walmart Associate Knowledge Copilot",
    page_icon="🏪",
    layout="wide"
)

@st.cache_resource
def load_rag_pipeline():
    rag = WalmartRAGPipeline()
    if not rag.load_index():
        with st.spinner("🚀 Setting up RAG pipeline... This may take a few minutes."):
            rag.setup_pipeline()
            rag.save_index()
    else:
        rag.load_models()
    return rag

def main():
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0071ce 0%, #004c91 100%); color: white; padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 2rem;">
        <h1>🏪 Walmart Associate Knowledge Copilot</h1>
        <p>Enterprise RAG System | Built for Production Scale</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Enterprise Demo")
        st.markdown("**Purpose:** Production-ready RAG system")
        st.markdown("**Scale:** 2.3M associates, 10,500 stores")
        
        st.markdown("### 💡 Try These Questions:")
        sample_questions = [
            "How do I apply for maternity leave?",
            "What's the overtime pay rate?",
            "Steps to request time off?",
            "Required documents for FMLA?",
            "How does scheduling work?"
        ]
        
        selected = st.selectbox("Sample queries:", ["Select..."] + sample_questions)
        
        st.markdown("### 🚀 Tech Stack")
        st.markdown("""
        - **🤖 Model**: FLAN-T5 + MiniLM
        - **🔍 Vector DB**: FAISS  
        - **⚡ Speed**: <2s response
        - **💾 Memory**: 8GB optimized
        - **🔒 Privacy**: Local deployment
        """)
    
    # Load pipeline
    try:
        rag = load_rag_pipeline()
        st.success("🟢 Knowledge Copilot Online")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()
    
    # Main interface
    st.markdown("## 💬 Ask Your Question")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "",
            placeholder="Ask about HR policies, procedures, training...",
            help="Get instant answers from official Walmart policies"
        )
    
    with col2:
        ask_button = st.button("🚀 Ask", type="primary")
    
    # Handle query
    if selected != "Select...":
        user_query = selected
        
    if user_query and (ask_button or selected != "Select..."):
        start_time = time.time()
        
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                result = rag.query(user_query)
                response_time = time.time() - start_time
                
                # Display answer
                st.markdown("### 📝 Response")
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #0071ce; margin: 1rem 0;">
                    <h4 style="color: #0071ce; margin-top: 0;">💡 {user_query}</h4>
                    <div style="font-size: 1.1rem; line-height: 1.6; color: #333;">
                        {result['answer']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⚡ Response Time", f"{response_time:.2f}s")
                with col2:
                    st.metric("📊 Sources Found", len(result['sources']))
                with col3:
                    st.metric("🎯 Avg Relevance", f"{sum(result['relevance_scores'])/len(result['relevance_scores']):.3f}")
                
                # Sources
                st.markdown("### 📚 Sources")
                for i, source in enumerate(set(result['sources'])):
                    score = result['relevance_scores'][i] if i < len(result['relevance_scores']) else 0
                    st.markdown(f"• **{source}** (Relevance: {score:.3f})")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>🏪 Walmart Knowledge Copilot</strong> | Built for Walmart ML Team Interview</p>
        <p>Demonstrates enterprise RAG architecture with production scalability</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
