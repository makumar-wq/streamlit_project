import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS
import time

# --- 1. PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Aid Analyst 8.0 (Comparison & Citations)")

# Initialize State
if 'storyboard_cart' not in st.session_state:
    st.session_state['storyboard_cart'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'ai_stage' not in st.session_state:
    st.session_state['ai_stage'] = "idle" # idle, confirming, researching, chatting

# --- 2. THE CITATION-AWARE RESEARCH AGENT ---
class ResearchAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = None
        if api_key:
            # Using Llama-3-8B-Instruct for reliable chat
            self.client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=api_key)

    def search_web_with_links(self, query):
        """Tool: Searches web and captures URLs."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No results.", []
            
            formatted_text = ""
            sources = []
            for r in results:
                formatted_text += f"- Title: {r['title']}\n  Snippet: {r['body']}\n  Link: {r['href']}\n\n"
                sources.append(r['href'])
                
            return formatted_text, sources
        except Exception as e:
            return f"Search Error: {e}", []

    def _ask_ai(self, messages, max_tokens=600):
        if not self.client:
            return "⚠️ API Token missing."
        try:
            response = self.client.chat_completion(messages, max_tokens=max_tokens)
            return response.choices[0].message.content
        except Exception as e:
            return f"AI Error: {e}"

    def generate_observation(self, data_context, x_axis, y_axis):
        """Step 1: Just read the graph."""
        prompt = f"""
        You are a Data Analyst. Look at this data summary:
        Metric: {y_axis} grouped by {x_axis}.
        Data: {data_context}
        
        Task: State the 3 most interesting fact, anomaly, or trend you see. 
        Be extremely brief (1 sentence). Do not explain "Why" yet.
        """
        return self._ask_ai([{"role": "user", "content": prompt}], max_tokens=100)

    def deep_research_with_sources(self, observation, context):
        """Step 2: Find the 'Why' with citations."""
        
        # 1. Generate Query
        q_prompt = f"Based on this observation: '{observation}', generate 1 specific search query to find the cause."
        query = self._ask_ai([{"role": "user", "content": q_prompt}], max_tokens=30).strip('"')
        
        # 2. Search
        web_data, links = self.search_web_with_links(query)
        
        # 3. Report
        final_prompt = f"""
        Observation: {observation}
        Web Evidence:
        {web_data}
        
        Task: Explain the CAUSE of the observation.
        Rules:
        1. Use Bullet Points.
        2. Be precise and to the point.
        3. CITE SOURCES immediately after the claim using [Source 1], [Source 2] format matching the links found.
        4. If evidence is weak, say so.
        """
        response = self._ask_ai([{"role": "user", "content": final_prompt}], max_tokens=500)
        
        # Append actual links at the bottom
        response += "\n\n**Sources:**\n"
        for i, link in enumerate(links):
            response += f"{i+1}. {link}\n"
            
        return response, query

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        # REPLACE WITH YOUR ACTUAL FILE
        df = pd.read_csv('merged_filtered.csv')
    except:
        st.error("File not found. Please upload 'merged_filtered.csv'.")
        st.stop()

    if 'Fiscal Year' in df.columns:
        df['Fiscal Year'] = pd.to_numeric(df['Fiscal Year'], errors='coerce').fillna(2025).astype(int)
    else:
        df['Fiscal Year'] = 2025
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Data Error: {e}")
    st.stop()

# --- 4. SIDEBAR ---
st.sidebar.title("🕵️ Analyst Workbench 8.0")

# Year Filter
all_years = sorted(df_raw['Fiscal Year'].unique())
c1, c2 = st.sidebar.columns(2)
start_year = c1.selectbox("From", all_years, index=0)
valid_ends = [y for y in all_years if y >= start_year]
end_year = c2.selectbox("To", valid_ends, index=len(valid_ends)-1)

# Universal Filter (ALL Columns)
all_columns = sorted(df_raw.columns.tolist()) # REQ 1: All columns available
filter_col = st.sidebar.selectbox("Filter Data By (Optional):", ["None"] + all_columns)

filter_values = []
if filter_col != "None":
    # Handle massive unique values gracefully
    unique_vals = sorted(df_raw[filter_col].astype(str).unique())
    if len(unique_vals) > 1000:
        st.sidebar.warning("Too many values to list. Type to search.")
    filter_values = st.sidebar.multiselect(f"Select values for {filter_col}:", unique_vals)

# API Key
st.sidebar.markdown("---")
hf_api_key = st.sidebar.text_input("🤖 AI Token (Free):", type="password")
agent = ResearchAgent(hf_api_key)

# --- 5. FILTERING ENGINE ---
mask = (df_raw['Fiscal Year'] >= start_year) & (df_raw['Fiscal Year'] <= end_year)
df_filtered = df_raw[mask]
if filter_col != "None" and filter_values:
    df_filtered = df_filtered[df_filtered[filter_col].astype(str).isin(filter_values)]

# --- 6. MAIN APP ---
tab_lab, tab_story = st.tabs(["🧪 Comparison Lab & AI", "📚 Final Storyboard"])

with tab_lab:
    st.subheader(f"Data Lab ({start_year}-{end_year})")
    st.caption(f"Analyzing {len(df_filtered):,} records")

    # --- A. CHART CONFIGURATOR (REQ 2: Comparison Tab) ---
    with st.container(border=True):
        st.write("🔧 **Chart Builder**")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"])
        with c2:
            # X-Axis: Allow any column
            x_axis = st.selectbox("X-Axis (Main Category)", all_columns, index=all_columns.index('Funding Agency Name') if 'Funding Agency Name' in all_columns else 0)
        with c3:
            # Y-Axis: Metric
            y_axis = st.selectbox("Y-Axis (Metric)", ["Record Count"] + all_columns, index=0)
        with c4:
            # REQ 2: The "Comparison" Feature (Color/Group)
            color_col = st.selectbox("Compare/Group By (Color)", ["None"] + all_columns, index=0)

        st.caption(f"ℹ️ **Explanation:** You are plotting **{x_axis}** vs **{y_axis}**. " + 
                   (f"Detailed by **{color_col}**." if color_col != "None" else ""))

    # --- B. PLOT GENERATION ---
    fig = None
    data_summary = ""

    # Logic to handle "Record Count" vs actual values
    if y_axis == "Record Count":
        if color_col == "None":
            # Simple Count
            df_plot = df_filtered[x_axis].value_counts().nlargest(15).reset_index()
            df_plot.columns = [x_axis, 'Count']
            fig = px.bar(df_plot, x=x_axis, y='Count', title=f"Top {x_axis}")
            data_summary = f"Top {x_axis} is {df_plot.iloc[0][x_axis]} with {df_plot.iloc[0]['Count']} records."
        else:
            # Stacked/Grouped Count (The Comparison)
            df_plot = df_filtered.groupby([x_axis, color_col]).size().reset_index(name='Count')
            # Optimization: Keep only top 15 X items to avoid crash
            top_x = df_plot.groupby(x_axis)['Count'].sum().nlargest(15).index
            df_plot = df_plot[df_plot[x_axis].isin(top_x)]
            
            fig = px.bar(df_plot, x=x_axis, y='Count', color=color_col, title=f"{x_axis} breakdown by {color_col}")
            data_summary = f"Comparison of {x_axis} grouped by {color_col}."
    
    elif chart_type == "Line Chart":
        # Time-Lapse logic
        if color_col == "None":
            df_plot = df_filtered.groupby(x_axis).size().reset_index(name='Count') # Fallback if X is time
            fig = px.line(df_plot, x=x_axis, y='Count', markers=True)
        else:
            df_plot = df_filtered.groupby([x_axis, color_col]).size().reset_index(name='Count')
            fig = px.line(df_plot, x=x_axis, y='Count', color=color_col, markers=True)
            
    # (Other chart types follow similar logic, simplified for brevity)
    else:
        # Default fallback
        fig = px.scatter(df_filtered, x=x_axis, y=y_axis, color=None if color_col=="None" else color_col)

    if fig:
        st.plotly_chart(fig, use_container_width=True)

        # --- C. AI RESEARCH WORKFLOW (REQ 3) ---
        st.markdown("---")
        st.subheader("🤖 AI Research Assistant")

        # Container for chat history
        chat_container = st.container(height=400)
        
        # 1. INITIAL TRIGGER
        col_btn, col_txt = st.columns([1, 4])
        with col_btn:
            if st.button("✨ Analyze Data"):
                st.session_state['ai_stage'] = "confirming"
                # Step 1: Read the graph
                obs = agent.generate_observation(data_summary, x_axis, y_axis)
                st.session_state['current_obs'] = obs
                # Add to history
                st.session_state['chat_history'].append({"role": "ai", "content": f"I see this in the data: **{obs}**\n\nShould I research the reasons behind this?"})

        # 2. CONFIRMATION & RESEARCH
        if st.session_state['ai_stage'] == "confirming":
            c1, c2 = st.columns(2)
            if c1.button("✅ Yes, Research it"):
                st.session_state['ai_stage'] = "researching"
                with st.spinner("Searching sources..."):
                    res, query = agent.deep_research_with_sources(st.session_state['current_obs'], data_summary)
                    st.session_state['chat_history'].append({"role": "ai", "content": res})
                    st.session_state['ai_stage'] = "chatting"
                    st.rerun()
            if c2.button("❌ No, Cancel"):
                st.session_state['ai_stage'] = "idle"
                st.session_state['chat_history'] = []
                st.rerun()

        # 3. DISPLAY CHAT HISTORY
        with chat_container:
            for msg in st.session_state['chat_history']:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # 4. FOLLOW-UP INPUT
        if st.session_state['ai_stage'] == "chatting":
            user_input = st.chat_input("Ask a follow-up question about this result...")
            if user_input:
                # Add user message
                st.session_state['chat_history'].append({"role": "user", "content": user_input})
                
                # Get AI response
                with st.spinner("Thinking..."):
                    # Contextual conversation
                    context = "\n".join([m["content"] for m in st.session_state['chat_history'][-3:]]) # Keep last 3 turns context
                    prompt = f"Context: {context}\nUser Question: {user_input}\nAnswer precisely."
                    answer = agent._ask_ai([{"role": "user", "content": prompt}])
                    st.session_state['chat_history'].append({"role": "ai", "content": answer})
                st.rerun()

        # --- D. ADD TO STORYBOARD ---
        with st.expander("🛒 Save to Storyboard"):
            # Get last AI message for notes
            default_note = ""
            if st.session_state['chat_history']:
                default_note = st.session_state['chat_history'][-1]['content']
                
            user_note = st.text_area("Notes:", value=default_note)
            if st.button("Add Chart & Notes"):
                st.session_state['storyboard_cart'].append({
                    'id': len(st.session_state['storyboard_cart']),
                    'fig': fig,
                    'title': f"{x_axis} vs {y_axis} ({start_year}-{end_year})",
                    'notes': user_note
                })
                st.success("Saved!")

# TAB 2 (Storyboard) - Standard display logic
with tab_story:
    st.header("📚 Final Report")
    if not st.session_state['storyboard_cart']:
        st.info("Empty.")
    else:
        for item in st.session_state['storyboard_cart']:
            st.markdown(f"### {item['title']}")
            st.plotly_chart(item['fig'])
            st.info(f"**Research:**\n\n{item['notes']}")
            st.markdown("---")