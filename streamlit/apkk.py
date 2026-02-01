import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS
import time

# --- 1. PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Aid StoryBuilder 7.0 (Deep Research)")

# Initialize State
if 'storyboard_cart' not in st.session_state:
    st.session_state['storyboard_cart'] = []

# --- 2. THE DEEP RESEARCH AGENT ---
class ResearchAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = None
        if api_key:
            # SWITCHING TO LLAMA-3 (More reliable on Free Tier for Chat)
            self.client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=api_key)

    def search_web(self, query):
        """Tool: Searches the real internet."""
        try:
            with DDGS() as ddgs:
                # Increased to 4 results for better breadth
                results = list(ddgs.text(query, max_results=4))
            if not results:
                return ""
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        except Exception as e:
            return f"Search Error: {e}"

    def _ask_ai(self, messages, max_tokens=600):
        """Helper to handle the Chat Completion API safely."""
        if not self.client:
            return "⚠️ Please enter a Hugging Face API Token in the sidebar."
        
        try:
            response = self.client.chat_completion(messages, max_tokens=max_tokens)
            return response.choices[0].message.content
        except Exception as e:
            return f"AI Error: {e}"

    def deep_research(self, data_context, chart_title):
        """
        Multi-Step Research Process:
        1. Brainstorm Search Queries (Economic, Political, etc.)
        2. Execute Multiple Searches
        3. Synthesize into a Story
        """
        status_box = st.status("🕵️ AI Agent Working...", expanded=True)
        
        # --- STEP 1: BRAINSTORMING ---
        status_box.write("🧠 Step 1: Analyzing data & brainstorming search angles...")
        brainstorm_prompt = f"""
        You are a Senior Data Investigator.
        
        DATA CONTEXT: {data_context}
        CHART: {chart_title}
        
        Task: Generate 3 distinct search queries to explain this trend.
        1. One finding the specific event/policy (e.g., "USAID budget cuts 2024").
        2. One looking for geopolitical context (e.g., "US relations with [Country] 2024").
        3. One checking for economic factors (e.g., "[Country] economic crisis 2024").
        
        Output format: Just the 3 queries, separated by newlines. No numbering.
        """
        
        queries_text = self._ask_ai([{"role": "user", "content": brainstorm_prompt}], max_tokens=100)
        queries = [q.strip() for q in queries_text.split('\n') if q.strip()][:3] # Take top 3
        
        # --- STEP 2: MULTI-THREADED SEARCHING ---
        all_findings = ""
        for i, q in enumerate(queries):
            status_box.write(f"🌍 Step 2.{i+1}: Searching: '{q}'...")
            res = self.search_web(q)
            all_findings += f"\n\n--- Findings for '{q}' ---\n{res}"
            time.sleep(1) # Polite delay
            
        # --- STEP 3: SYNTHESIS & STORYTELLING ---
        status_box.write("📝 Step 3: Connecting the dots & writing story...")
        final_prompt = f"""
        You are an Investigative Journalist. Write a data story based on this evidence.
        
        THE DATA (What happened):
        {data_context}
        
        THE EVIDENCE (Why it happened):
        {all_findings}
        
        TASK:
        Write a cohesive narrative explaining the "Why".
        - Start with a strong headline.
        - Connect the data drop/rise directly to the real-world events found.
        - Mention specific policies, conflicts, or economic events found in the search.
        - If evidence is conflicting, mention that.
        """
        
        final_story = self._ask_ai([{"role": "user", "content": final_prompt}], max_tokens=800)
        
        status_box.update(label="✅ Research Complete!", state="complete", expanded=False)
        return final_story, queries

    def synthesize_report(self, all_notes):
        """Final Storyboard Synthesis"""
        prompt = f"""
        Act as a Lead Strategy Officer.
        Synthesize these separate research notes into one Final Executive Report.
        
        INPUT NOTES:
        {all_notes}
        
        OUTPUT:
        1. Executive Summary (The Big Picture)
        2. Key Drivers (Bullet points of main causes)
        3. Strategic Recommendations
        """
        return self._ask_ai([{"role": "user", "content": prompt}], max_tokens=1000)

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        # REPLACE WITH YOUR ACTUAL FILE NAME
        df = pd.read_csv('merged_filtered.csv')
    except FileNotFoundError:
        st.error("File 'merged_filtered.csv' not found. Please upload it.")
        st.stop()

    if 'Fiscal Year' in df.columns:
        df['Fiscal Year'] = pd.to_numeric(df['Fiscal Year'], errors='coerce').fillna(2025).astype(int)
    else:
        df['Fiscal Year'] = 2025
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Data Load Error: {e}")
    st.stop()

# --- 4. SIDEBAR ---
st.sidebar.title("🕵️ Analyst Workbench 7.0")

# Flight-Style Date Picker
all_years = sorted(df_raw['Fiscal Year'].unique())
if not all_years:
    st.error("No years found in dataset.")
    st.stop()

c1, c2 = st.sidebar.columns(2)
start_year = c1.selectbox("From", all_years, index=0)
valid_ends = [y for y in all_years if y >= start_year]
end_year = c2.selectbox("To", valid_ends, index=len(valid_ends)-1)

# Universal Filter
filter_col = st.sidebar.selectbox("Filter By:", ["None"] + sorted([c for c in df_raw.columns if df_raw[c].dtype == 'object']))
filter_values = []
if filter_col != "None":
    filter_values = st.sidebar.multiselect(f"Select {filter_col}:", sorted(df_raw[filter_col].astype(str).unique()))

# API Key
st.sidebar.markdown("---")
hf_api_key = st.sidebar.text_input("🤖 AI Token (Free):", type="password", help="Hugging Face Token")
agent = ResearchAgent(hf_api_key)

# --- 5. FILTERING ---
mask = (df_raw['Fiscal Year'] >= start_year) & (df_raw['Fiscal Year'] <= end_year)
df_filtered = df_raw[mask]
if filter_col != "None" and filter_values:
    df_filtered = df_filtered[df_filtered[filter_col].astype(str).isin(filter_values)]

# --- 6. MAIN APP ---
tab_lab, tab_story = st.tabs(["🧪 Research Lab", "📚 Storyboard"])

with tab_lab:
    st.subheader(f"Data Lab ({start_year}-{end_year})")
    st.caption(f"Analyzing {len(df_filtered):,} records")
    
    # Chart Controls
    with st.container(border=True):
        cc1, cc2, cc3 = st.columns(3)
        chart_type = cc1.selectbox("Chart Type", ["Line (Time-Lapse)", "Bar (Compare)", "Sankey (Flow)"])
        cat_cols = [c for c in df_filtered.columns if df_filtered[c].dtype == 'object']
        x_axis = cc2.selectbox("X-Axis", cat_cols, index=0)
        y_axis = cc3.selectbox("Y-Axis", ["Record Count"] + cat_cols, index=0)

    # Plot Generation
    fig = None
    data_narrative = "" 

    if chart_type == "Line (Time-Lapse)":
        if start_year == end_year:
            st.warning("⚠️ Time-Lapse requires a year range > 1 year.")
        else:
            trend_df = df_filtered.groupby(['Fiscal Year', x_axis]).size().reset_index(name='Count')
            top_items = trend_df.groupby(x_axis)['Count'].sum().nlargest(5).index
            trend_df = trend_df[trend_df[x_axis].isin(top_items)]
            
            fig = px.line(trend_df, x='Fiscal Year', y='Count', color=x_axis, markers=True, title=f"{x_axis} Trend")
            
            start_v = trend_df[trend_df['Fiscal Year'] == start_year]['Count'].sum()
            end_v = trend_df[trend_df['Fiscal Year'] == end_year]['Count'].sum()
            trend_direction = "increased" if end_v > start_v else "decreased"
            data_narrative = f"The data shows a trend for {x_axis}. The total count {trend_direction} from {start_v} in {start_year} to {end_v} in {end_year}."

    elif chart_type == "Bar (Compare)":
        if y_axis == "Record Count":
            counts = df_filtered[x_axis].value_counts().nlargest(10)
            fig = px.bar(x=counts.values, y=counts.index, orientation='h', title=f"Top {x_axis}")
            data_narrative = f"The bar chart highlights {counts.idxmax()} as the dominant {x_axis}, with {counts.max()} records, significantly higher than others."
        else:
            mat = df_filtered.groupby([x_axis, y_axis]).size().reset_index(name='Count')
            fig = px.bar(mat, x=x_axis, y='Count', color=y_axis, title=f"{x_axis} vs {y_axis}")
            data_narrative = f"The chart compares {x_axis} broken down by {y_axis}."

    elif chart_type == "Sankey (Flow)":
        if y_axis == "Record Count":
            st.error("For Sankey, please select a Category for Y-Axis.")
        else:
            sankey_data = df_filtered.groupby([x_axis, y_axis]).size().reset_index(name='Count')
            top_x = sankey_data.groupby(x_axis)['Count'].sum().nlargest(10).index
            top_y = sankey_data.groupby(y_axis)['Count'].sum().nlargest(10).index
            sankey_f = sankey_data[(sankey_data[x_axis].isin(top_x)) & (sankey_data[y_axis].isin(top_y))]
            
            all_nodes = list(pd.concat([sankey_f[x_axis], sankey_f[y_axis]]).unique())
            node_map = {node: i for i, node in enumerate(all_nodes)}
            fig = go.Figure(data=[go.Sankey(
                node=dict(label=all_nodes, pad=15, thickness=20, color="blue"),
                link=dict(
                    source=sankey_f[x_axis].map(node_map),
                    target=sankey_f[y_axis].map(node_map),
                    value=sankey_f['Count']
                )
            )])
            fig.update_layout(title=f"Flow: {x_axis} ➔ {y_axis}")
            data_narrative = f"A Sankey diagram showing the flow of resources from {x_axis} to {y_axis}."

    if fig:
        st.plotly_chart(fig, use_container_width=True)

        # --- THE AI RESEARCH INTERFACE ---
        st.markdown("### 🤖 Deep Research Agent")
        c_btn, c_res = st.columns([1, 4])
        
        with c_btn:
            run_research = st.button("✨ Deep Research 'Why'")
        
        if run_research:
            # We don't need st.spinner here because the status box inside deep_research handles the UI
            insight, queries_used = agent.deep_research(data_narrative, f"{x_axis} Analysis")
            st.session_state['current_insight'] = insight
            st.session_state['last_queries'] = queries_used
        
        if 'current_insight' in st.session_state:
            with st.expander("See Search Queries Used"):
                st.write(st.session_state.get('last_queries'))
            st.success(f"**Analyst Story:**\n\n{st.session_state['current_insight']}")

        # Add to Storyboard
        with st.form("save_cart"):
            note_val = st.session_state.get('current_insight', "")
            user_notes = st.text_area("Notes for Final Report:", value=note_val, height=200)
            if st.form_submit_button("🛒 Add to Storyboard"):
                st.session_state['storyboard_cart'].append({
                    'id': len(st.session_state['storyboard_cart']) + 1,
                    'fig': fig,
                    'title': f"{x_axis} ({start_year}-{end_year})",
                    'year_range': f"{start_year}-{end_year}",
                    'filters': f"{filter_col}={filter_values}" if filter_col != "None" else "Global",
                    'notes': user_notes
                })
                st.success("Added!")

# TAB 2 (Storyboard)
with tab_story:
    st.header("📚 Executive Analysis Report")
    
    if not st.session_state['storyboard_cart']:
        st.info("Your storyboard is empty. Generate charts in the Lab and add them here.")
    else:
        for item in st.session_state['storyboard_cart']:
            with st.container(border=True):
                c_meta, c_chart = st.columns([1, 3])
                with c_meta:
                    st.subheader(f"#{item['id']}")
                    st.caption(f"📅 {item['year_range']}")
                    st.caption(f"🔍 {item['filters']}")
                    with st.expander("Read Notes"):
                        st.write(item['notes'])
                    if st.button("🗑️ Remove", key=f"del_{item['id']}"):
                        st.session_state['storyboard_cart'].remove(item)
                        st.rerun()
                with c_chart:
                    st.markdown(f"### {item['title']}")
                    st.plotly_chart(item['fig'], use_container_width=True)

        st.markdown("---")
        st.header("🏁 Final Story Synthesis")
        c_syn_gen, c_syn_out = st.columns([1, 2])
        
        with c_syn_gen:
            if st.button("🤖 Generate Final Story (AI)"):
                all_notes = "\n".join([f"- {i['title']}: {i['notes']}" for i in st.session_state['storyboard_cart']])
                with st.spinner("Synthesizing all research..."):
                    final_story = agent.synthesize_report(all_notes)
                    st.session_state['final_story'] = final_story
        
        with c_syn_out:
            final_text = st.text_area("Final Executive Summary", value=st.session_state.get('final_story', ''), height=500)
            if st.button("🖨️ Export Report"):
                st.balloons()
                st.success("Report exported!")