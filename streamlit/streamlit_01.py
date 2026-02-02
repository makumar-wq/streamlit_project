"""
Data Story Builder - Professional Data Analysis Framework
Version: 3.0

A comprehensive tool for comparing any dimensions in your dataset,
discovering patterns across time periods, and building data-driven narratives.

Features:
- Multi-dimensional comparison with any columns
- Multiple visualization types (Bar, Line, Area, Scatter, Heatmap)
- Year-by-year side-by-side analysis
- Smart entity filtering with impact preview
- AI-powered analysis with real data context
- Story building and export

Author: Data Story Builder Framework
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional AI imports
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    layout="wide",
    page_title="Data Story Builder",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Main Theme & Reset */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #020617 90%);
        font-family: 'Outfit', sans-serif;
        color: #e2e8f0;
    }
    
    /* Headers */
    h1, h2, h3, .main-header, .sub-header {
        font-family: 'Outfit', sans-serif !important;
        color: #f8fafc !important;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.3);
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-bottom: 2.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        padding-bottom: 1rem;
    }
    
    /* Cards & Containers */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        border-color: rgba(56, 189, 248, 0.5);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        text-shadow: 0 0 20px rgba(248, 250, 252, 0.2);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Info Boxes - Professional Look */
    .filter-impact-box {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #fcd34d;
    }
    .data-context-box {
        background: rgba(56, 189, 248, 0.1);
        border-left: 4px solid #38bdf8;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 12px 12px 0;
        color: #e0f2fe;
    }
    .finding-card {
        background: rgba(16, 185, 129, 0.05);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #10b981;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        color: #e2e8f0;
        transition: all 0.2s;
    }
    .finding-card:hover {
        background: rgba(16, 185, 129, 0.1);
        transform: translateX(5px);
    }
    
    .year-badge {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Explanation & Code */
    .explanation-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
    .explanation-box code {
        background: rgba(15, 23, 42, 0.8);
        color: #fb7185;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        border: 1px solid rgba(251, 113, 133, 0.2);
    }
    
    /* Streamlit Components Override */
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
    }
    .stSelectbox > div > div {
        background-color: #0f172a !important;
        color: white !important;
    }
    .stTextInput > div > div {
        background-color: #0f172a !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
def init_state():
    defaults = {
        'story_sections': [],
        'current_analysis': None,
        'combined_story': '',
        'filter_active': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data(file):
    """Load data from various file formats"""
    if file is None:
        return None
    try:
        name = file.name.lower()
        if name.endswith('.gz'):
            return pd.read_csv(file, compression='gzip')
        elif name.endswith('.csv'):
            return pd.read_csv(file)
        elif name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file)
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# =============================================================================
# DATA INTELLIGENCE
# =============================================================================
class DataIntelligence:
    """Analyze and classify dataset columns"""
    
    def __init__(self, df):
        self.df = df
        self.classifications = self._classify_all()
    
    def _classify_all(self):
        """Classify each column by type"""
        result = {}
        for col in self.df.columns:
            col_lower = col.lower()
            dtype = self.df[col].dtype
            nunique = self.df[col].nunique()
            
            if any(t in col_lower for t in ['year', 'date', 'month', 'quarter', 'period']):
                ctype = 'time'
            elif col_lower.endswith(' id') or col_lower == 'id':
                ctype = 'id'
            elif pd.api.types.is_numeric_dtype(dtype) and nunique > 20:
                ctype = 'numeric'
            elif nunique < 1000:
                ctype = 'categorical'
            else:
                ctype = 'text'
            
            result[col] = {
                'type': ctype,
                'nunique': nunique,
                'dtype': str(dtype)
            }
        return result
    
    def get_by_type(self, ctype):
        return [c for c, info in self.classifications.items() if info['type'] == ctype]
    
    def get_analysis_columns(self):
        """Columns suitable for analysis (exclude IDs)"""
        return [c for c, info in self.classifications.items() if info['type'] != 'id']
    
    def get_numeric_columns(self):
        return self.get_by_type('numeric')
    
    def get_time_columns(self):
        return self.get_by_type('time')

# =============================================================================
# COMPARISON ENGINE
# =============================================================================
class ComparisonEngine:
    """Core engine for data comparison and aggregation"""
    
    def __init__(self, df, time_col=None, metric_col=None):
        self.df_original = df
        self.time_col = time_col
        self.metric_col = metric_col
    
    def apply_filters(self, df, filters):
        """Apply multiple filters to dataframe"""
        df_filtered = df.copy()
        for f in filters:
            if f.get('column') and f.get('values'):
                df_filtered = df_filtered[df_filtered[f['column']].astype(str).isin([str(v) for v in f['values']])]
        return df_filtered
    
    def aggregate_data(self, df, group_cols, top_n=10, primary_col=None):
        """Aggregate data by specified columns"""
        
        if self.metric_col and self.metric_col in df.columns:
            agg_df = df.groupby(group_cols, as_index=False)[self.metric_col].sum()
            agg_df = agg_df.rename(columns={self.metric_col: 'Value'})
        else:
            agg_df = df.groupby(group_cols, as_index=False).size()
            agg_df = agg_df.rename(columns={'size': 'Value'})
        
        # Get top N for primary column
        if primary_col and primary_col in group_cols:
            top_entities = agg_df.groupby(primary_col)['Value'].sum().nlargest(top_n).index.tolist()
            agg_df = agg_df[agg_df[primary_col].isin(top_entities)]
            return agg_df, top_entities
        
        return agg_df, []
    
    def get_comparison_data(self, col1, col2, years, filters=None, top_n=10):
        """Get aggregated comparison data"""
        
        df = self.df_original.copy()
        
        # Apply filters
        if filters:
            df = self.apply_filters(df, filters)
        
        # Filter years
        if self.time_col and years:
            df = df[df[self.time_col].isin(years)]
        
        if df.empty:
            return pd.DataFrame(), [], {}
        
        # Build group columns
        group_cols = [col1, col2]
        if self.time_col:
            group_cols = [self.time_col] + group_cols
        
        # Aggregate
        agg_df, top_entities = self.aggregate_data(df, group_cols, top_n, col1)
        
        # Calculate stats
        stats = self._calculate_stats(agg_df, col1, col2)
        
        return agg_df, top_entities, stats
    
    def _calculate_stats(self, agg_df, col1, col2):
        """Calculate summary statistics"""
        if agg_df.empty:
            return {}
        
        total = agg_df['Value'].sum()
        
        col1_totals = agg_df.groupby(col1)['Value'].sum().sort_values(ascending=False)
        col2_totals = agg_df.groupby(col2)['Value'].sum().sort_values(ascending=False)
        
        stats = {
            'total': total,
            'n_col1': agg_df[col1].nunique(),
            'n_col2': agg_df[col2].nunique(),
            'top_col1': col1_totals.head(5).to_dict(),
            'top_col2': col2_totals.head(5).to_dict(),
            'concentration_top3': col1_totals.head(3).sum() / total * 100 if total > 0 else 0
        }
        
        # Top combination
        combo = agg_df.groupby([col1, col2])['Value'].sum()
        if not combo.empty:
            top_idx = combo.idxmax()
            stats['top_combo'] = {'entities': top_idx, 'value': combo.max()}
        
        return stats
    
    def get_filter_impact(self, filters):
        """Show how filters affect the data"""
        original_count = len(self.df_original)
        filtered_df = self.apply_filters(self.df_original, filters)
        filtered_count = len(filtered_df)
        
        impact = {
            'original': original_count,
            'filtered': filtered_count,
            'removed': original_count - filtered_count,
            'pct_remaining': (filtered_count / original_count * 100) if original_count > 0 else 0
        }
        
        # What's unique in filtered data
        if filters and filters[0].get('column'):
            for f in filters:
                col = f['column']
                if col in filtered_df.columns:
                    impact[f'unique_{col}'] = filtered_df[col].nunique()
        
        return impact, filtered_df

# =============================================================================
# VISUALIZATION ENGINE
# =============================================================================
class VisualizationEngine:
    """Create various chart types"""
    
    CHART_TYPES = ['Grouped Bar', 'Stacked Bar', 'Line', 'Area', 'Heatmap', 'Treemap']
    COLOR_SCALES = ['Cyber', 'Neon', 'Pastel', 'Vivid', 'Plotly']
    
    # Custom Palettes
    THEMES = {
        'Cyber': ['#00f2ff', '#ff00e5', '#39ff14', '#fff01f', '#f7f7f7', '#aa00ff'],
        'Neon': ['#f72585', '#b5179e', '#7209b7', '#560bad', '#480ca8', '#3a0ca3'],
    }
    
    def __init__(self, color_scale='Cyber'):
        if color_scale in self.THEMES:
            self.colors = self.THEMES[color_scale]
        else:
            self.colors = getattr(px.colors.qualitative, color_scale, px.colors.qualitative.D3)
        self.template = 'plotly_dark'
    
    def create_chart(self, df, x_col, y_col, color_col, chart_type, title=""):
        """Create chart based on type selection"""
        
        if df.empty:
            return self._empty_chart(title)
        
        chart_funcs = {
            'Grouped Bar': self._grouped_bar,
            'Stacked Bar': self._stacked_bar,
            'Line': self._line_chart,
            'Area': self._area_chart,
            'Heatmap': self._heatmap,
            'Treemap': self._treemap
        }
        
        func = chart_funcs.get(chart_type, self._grouped_bar)
        return func(df, x_col, y_col, color_col, title)
    
    def _empty_chart(self, title):
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title, height=350)
        return fig
    
    def _grouped_bar(self, df, x_col, y_col, color_col, title):
        fig = px.bar(
            df, x=x_col, y=y_col, color=color_col,
            barmode='group', title=title,
            color_discrete_sequence=self.colors
        )
        self._apply_layout(fig)
        return fig
    
    def _stacked_bar(self, df, x_col, y_col, color_col, title):
        fig = px.bar(
            df, x=x_col, y=y_col, color=color_col,
            barmode='stack', title=title,
            color_discrete_sequence=self.colors
        )
        self._apply_layout(fig)
        return fig
    
    def _line_chart(self, df, x_col, y_col, color_col, title):
        # Aggregate for line chart
        line_df = df.groupby([x_col, color_col])[y_col].sum().reset_index()
        fig = px.line(
            line_df, x=x_col, y=y_col, color=color_col,
            markers=True, title=title,
            color_discrete_sequence=self.colors
        )
        self._apply_layout(fig)
        return fig
    
    def _area_chart(self, df, x_col, y_col, color_col, title):
        area_df = df.groupby([x_col, color_col])[y_col].sum().reset_index()
        fig = px.area(
            area_df, x=x_col, y=y_col, color=color_col,
            title=title, color_discrete_sequence=self.colors
        )
        self._apply_layout(fig)
        return fig
    
    def _heatmap(self, df, x_col, y_col, color_col, title):
        # Pivot for heatmap
        try:
            pivot = df.groupby([x_col, color_col])[y_col].sum().reset_index()
            pivot_wide = pivot.pivot(index=x_col, columns=color_col, values=y_col).fillna(0)
            
            # Relaxed constraint: Allow any shape as long as data exists
            if pivot_wide.empty:
                return self._empty_chart(f"{title} (No matching data)")
            
            fig = px.imshow(
                pivot_wide, title=title,
                color_continuous_scale='Plasma',
                aspect='auto'
            )
            fig.update_layout(height=max(350, pivot_wide.shape[0] * 25))
            return fig
        except Exception as e:
            return self._empty_chart(f"{title} (Error: {str(e)[:50]})")
    
    def _treemap(self, df, x_col, y_col, color_col, title):
        try:
            tree_df = df.groupby([x_col, color_col])[y_col].sum().reset_index()
            fig = px.treemap(
                tree_df, path=[x_col, color_col], values=y_col,
                title=title, color_discrete_sequence=self.colors
            )
            fig.update_layout(height=450)
            return fig
        except:
            return self._empty_chart(f"{title} (Treemap error)")
    
    def _apply_layout(self, fig):
        """Apply consistent professional layout"""
        fig.update_layout(
            template=self.template,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Outfit', size=12, color='#e2e8f0'),
            height=400,
            margin=dict(l=40, r=40, t=50, b=80),
            xaxis=dict(
                showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)',
                tickfont=dict(color='#94a3b8')
            ),
            yaxis=dict(
                showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)',
                tickfont=dict(color='#94a3b8'),
                title_font=dict(size=12, color='#cbd5e1')
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=11, color='#cbd5e1')
            ),
            title=dict(
                font=dict(size=16, color='#f8fafc', family='Outfit')
            )
        )
    
    def create_trend_chart(self, df, time_col, group_col, value_col='Value'):
        """Create trend line chart with robustness checks"""
        if df.empty:
            return self._empty_chart("Trend Chart (No Data)")
        
        # Robust aggregation
        trend_df = df.groupby([time_col, group_col])[value_col].sum().reset_index()
        
        # Check if we have enough points for a line
        unique_times = trend_df[time_col].nunique()
        if unique_times < 2:
            # Fallback to bar if only 1 time point
            fig = px.bar(
                trend_df, x=time_col, y=value_col, color=group_col,
                title=f"Trend: {group_col} (Single Period)",
                barmode='group',
                color_discrete_sequence=self.colors
            )
        else:
            fig = px.line(
                trend_df, x=time_col, y=value_col, color=group_col,
                markers=True, title=f"Trend: {group_col} over {time_col}",
                color_discrete_sequence=self.colors,
                line_shape='spline',
                render_mode='svg'
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')), line=dict(width=3))
        
        self._apply_layout(fig)
        return fig
    
    def create_summary_heatmap(self, df, row_col, col_col, value_col='Value'):
        """Create relationship heatmap"""
        try:
            pivot = df.groupby([row_col, col_col])[value_col].sum().reset_index()
            pivot_wide = pivot.pivot(index=row_col, columns=col_col, values=value_col).fillna(0)
            
            if pivot_wide.empty:
                return self._empty_chart("Relationship Map (No Data)")
            
            fig = px.imshow(
                pivot_wide,
                title=f"Relationship: {row_col} × {col_col}",
                color_continuous_scale='Plasma',
                aspect='auto'
            )
            fig.update_layout(height=max(400, pivot_wide.shape[0] * 22))
            return fig
        except Exception as e:
            return self._empty_chart(f"Heatmap Error: {str(e)[:40]}")

    def create_correlation_matrix(self, df):
        """Create correlation matrix for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return self._empty_chart("Correlation (Not enough numeric columns)")
            
        corr = numeric_df.corr().round(2)
        
        fig = px.imshow(
            corr, 
            text_auto=True,
            title="Correlation Matrix (Pearson)",
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            aspect='auto'
        )
        self._apply_layout(fig)
        return fig

    def create_distribution_plot(self, df, x_col, y_col=None):
        """Create distribution analysis (Box/Violin)"""
        # If y_col is None (Record Count), we can't plot distribution of value
        # But we can plot distribution of counts? 
        # Better: if y_col is present, plot box of y_col by x_col
        
        if y_col and y_col != 'Value' and y_col != 'Record Count':
             fig = px.box(
                 df, x=x_col, y=y_col, color=x_col,
                 title=f"Distribution of {y_col} by {x_col}",
                 color_discrete_sequence=self.colors
             )
        else:
            # Histogram of categories
            fig = px.histogram(
                df, x=x_col, 
                title=f"Distribution of {x_col} Occurrences",
                color=x_col,
                color_discrete_sequence=self.colors
            )
            
        self._apply_layout(fig)
        return fig

# =============================================================================
# FINDINGS GENERATOR
# =============================================================================
def generate_findings(stats, col1, col2, metric_name, time_col=None, years=None, agg_df=None):
    """Generate human-readable findings from statistics"""
    findings = []
    
    if not stats:
        return ["No data available for analysis"]
    
    total = stats.get('total', 0)
    if total == 0:
        return ["No records found with current filters"]
    
    # Format value based on magnitude
    def fmt(v):
        if v >= 1e9:
            return f"${v/1e9:.2f}B"
        elif v >= 1e6:
            return f"${v/1e6:.1f}M"
        elif v >= 1e3:
            return f"{v/1e3:.1f}K"
        else:
            return f"{v:,.0f}"
    
    # Finding 1: Leader
    if stats.get('top_col1'):
        leader = list(stats['top_col1'].items())[0]
        pct = leader[1] / total * 100 if total > 0 else 0
        findings.append(f"**{leader[0]}** leads with {fmt(leader[1])} ({pct:.1f}% of total {metric_name})")
    
    # Finding 2: Top combination
    if stats.get('top_combo'):
        tc = stats['top_combo']
        findings.append(f"Strongest combination: **{tc['entities'][0]}** + **{tc['entities'][1]}** = {fmt(tc['value'])}")
    
    # Finding 3: Concentration
    conc = stats.get('concentration_top3', 0)
    if conc > 60:
        findings.append(f"High concentration: Top 3 {col1} entities control **{conc:.0f}%** of total")
    elif conc < 40:
        findings.append(f"Distributed: Top 3 {col1} entities account for only **{conc:.0f}%**")
    
    # Finding 4: Secondary dimension
    if stats.get('top_col2'):
        top_c2 = list(stats['top_col2'].items())[0]
        pct_c2 = top_c2[1] / total * 100 if total > 0 else 0
        findings.append(f"In **{col2}**, '{top_c2[0]}' dominates at **{pct_c2:.1f}%**")
    
    # Finding 5: Trend (if time data available)
    if time_col and agg_df is not None and not agg_df.empty and time_col in agg_df.columns:
        yearly = agg_df.groupby(time_col)['Value'].sum().sort_index()
        if len(yearly) >= 2:
            first, last = yearly.iloc[0], yearly.iloc[-1]
            if first > 0:
                change = (last - first) / first * 100
                direction = "increased" if change > 0 else "decreased"
                findings.append(f"Trend: Total {direction} by **{abs(change):.1f}%** from {yearly.index[0]} to {yearly.index[-1]}")
    
    return findings

# =============================================================================
# AI AGENT
# =============================================================================
class AIAgent:
    """AI-powered analysis with web research"""
    
    def __init__(self, api_key=None):
        self.client = None
        if api_key and HF_AVAILABLE:
            try:
                self.client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=api_key)
            except:
                pass
    
    def _call(self, prompt, max_tokens=800):
        if not self.client:
            return None
        try:
            resp = self.client.chat_completion([{"role": "user", "content": prompt}], max_tokens=max_tokens)
            return resp.choices[0].message.content
        except:
            return None
    
    def analyze(self, data_summary, config):
        """Generate comprehensive analysis"""
        prompt = f"""You are an expert data analyst. Analyze this data comparison.

CONTEXT:
- Comparing: {config['col1']} vs {config['col2']}
- Time: {config.get('years', 'N/A')}
- Metric: {config.get('metric', 'Count')}
- Filters: {config.get('filters', 'None')}

DATA:
{data_summary}

Provide analysis with:
1. KEY PATTERNS (3-5 bullet points with specific numbers)
2. NOTABLE TRENDS (what's changing over time?)
3. TOP PERFORMERS (who dominates?)
4. INSIGHTS (what questions does this raise?)

Be specific. Use exact numbers from the data. No generic statements."""

        result = self._call(prompt, 1000)
        return result if result else self._fallback_analysis(data_summary, config)
    
    def _fallback_analysis(self, data_summary, config):
        return f"""### Data Analysis Summary

**Comparison:** {config['col1']} vs {config['col2']}
**Metric:** {config.get('metric', 'Record Count')}

{data_summary}

*For AI-powered insights, add a HuggingFace API token in the sidebar.*"""
    
    def search_web(self, query):
        """Search web for context"""
        if not DDGS_AVAILABLE:
            return [], ""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=4))
            sources = [{'title': r.get('title', ''), 'url': r.get('href', ''), 'snippet': r.get('body', '')[:250]} for r in results]
            context = "\n".join([f"[{i+1}] {s['title']}: {s['snippet']}" for i, s in enumerate(sources)])
            return sources, context
        except:
            return [], ""
    
    def explain_finding(self, finding, sources_context):
        """Explain a finding using web sources"""
        if not sources_context:
            return "No web sources found."
        
        prompt = f"""Finding: {finding}

Web Research:
{sources_context}

Explain the likely causes behind this finding. Cite sources as [1], [2], etc. Keep under 150 words."""
        
        result = self._call(prompt, 300)
        return result if result else "Could not generate explanation."
    
    def combine_story(self, sections):
        """Combine multiple sections into a narrative"""
        sections_text = "\n\n---\n\n".join([f"## {s['title']}\n{chr(10).join(s['findings'])}" for s in sections])
        
        prompt = f"""Create a unified data story from these sections:

{sections_text}

Write a cohesive report (400-600 words) with:
- Compelling headline
- Executive summary
- Key findings as narrative
- Conclusions"""
        
        result = self._call(prompt, 1200)
        return result if result else f"# Data Story\n\n{sections_text}"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def format_number(v, is_currency=False):
    """Format numbers for display"""
    if pd.isna(v):
        return "N/A"
    if is_currency:
        if v >= 1e9:
            return f"${v/1e9:.2f}B"
        elif v >= 1e6:
            return f"${v/1e6:.1f}M"
        else:
            return f"${v:,.0f}"
    else:
        if v >= 1e6:
            return f"{v/1e6:.1f}M"
        elif v >= 1e3:
            return f"{v/1e3:.1f}K"
        else:
            return f"{v:,.0f}"

def generate_data_summary(agg_df, col1, col2, time_col, stats, years, filters=None):
    """Generate text summary for AI"""
    lines = []
    lines.append("=" * 50)
    lines.append(f"COMPARISON: {col1} vs {col2}")
    lines.append(f"YEARS: {min(years)} to {max(years)}" if years else "All time")
    if filters:
        for f in filters:
            if f.get('column') and f.get('values'):
                lines.append(f"FILTER: {f['column']} = {', '.join(map(str, f['values']))}")
    lines.append("=" * 50)
    
    lines.append(f"\nTOTAL VALUE: {format_number(stats.get('total', 0), True)}")
    lines.append(f"Unique {col1}: {stats.get('n_col1', 0)}")
    lines.append(f"Unique {col2}: {stats.get('n_col2', 0)}")
    
    if stats.get('top_combo'):
        tc = stats['top_combo']
        lines.append(f"Top combination: {tc['entities'][0]} + {tc['entities'][1]} = {format_number(tc['value'], True)}")
    
    lines.append(f"\n--- TOP {col1} ---")
    for entity, val in list(stats.get('top_col1', {}).items())[:5]:
        lines.append(f"  {entity}: {format_number(val, True)}")
    
    lines.append(f"\n--- {col2} DISTRIBUTION ---")
    for cat, val in list(stats.get('top_col2', {}).items())[:5]:
        lines.append(f"  {cat}: {format_number(val, True)}")
    
    if time_col and not agg_df.empty:
        lines.append(f"\n--- BY YEAR ---")
        for year in sorted(agg_df[time_col].unique()):
            year_total = agg_df[agg_df[time_col] == year]['Value'].sum()
            lines.append(f"  {year}: {format_number(year_total, True)}")
    
    return "\n".join(lines)

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    st.markdown('<p class="main-header">📊 Data Story Builder</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Compare any dimensions • Discover patterns • Build narratives</p>', unsafe_allow_html=True)
    
    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Data Upload
        st.subheader("📁 Data")
        uploaded = st.file_uploader("Upload dataset", type=['csv', 'gz', 'xlsx', 'xls'])
        
        df = load_data(uploaded)
        if df is None:
            st.warning("Please upload a dataset")
            st.stop()
        
        st.success(f"✅ {len(df):,} rows × {len(df.columns)} cols")
        
        # Data Intelligence
        intel = DataIntelligence(df)
        
        # Sequence / Time Axis
        st.subheader("⏱️ Sequence / Time Axis")
        
        # Adaptive column selection
        suggested_time = intel.get_time_columns()
        use_all_cols = st.checkbox("Show non-time columns", value=False, help="Use arbitrary events/categories as sequence")
        
        time_options = ['(None)'] + (df.columns.tolist() if use_all_cols else suggested_time)
        default_idx = 1 if suggested_time and not use_all_cols else 0
        
        time_col = st.selectbox("Select Axis", time_options, index=default_idx)
        
        selected_years = None
        if time_col != '(None)':
            # Check if numeric/time-like for Range selection
            is_numeric_seq = pd.api.types.is_numeric_dtype(df[time_col])
            
            if is_numeric_seq:
                # Handle numeric years/periods
                df[time_col] = pd.to_numeric(df[time_col], errors='coerce').fillna(0).astype(int)
                all_years = sorted([y for y in df[time_col].unique() if y > 0])
                
                if all_years:
                    c1, c2 = st.columns(2)
                    start_yr = c1.selectbox("From", all_years, index=0)
                    valid_ends = [y for y in all_years if y >= start_yr]
                    end_yr = c2.selectbox("To", valid_ends, index=len(valid_ends)-1 if valid_ends else 0)
                    selected_years = [y for y in all_years if start_yr <= y <= end_yr]
            else:
                # Handle categorical events/sequences
                st.caption(f"Using **{time_col}** as categorical sequence.")
                # Preserve order validation if possible, otherwise use appearance
                selected_years = df[time_col].unique().tolist()
        else:
            time_col = None
        
        # Metric
        st.subheader("📏 Metric")
        numeric_cols = intel.get_numeric_columns()
        metric_options = ['Record Count'] + numeric_cols
        metric_choice = st.selectbox("Aggregate By", metric_options)
        metric_col = None if metric_choice == 'Record Count' else metric_choice
        
        # Explain metric
        st.caption("""
        **Record Count** = Number of rows/transactions
        **Dollar columns** = Sum of monetary values
        """)
        
        # AI
        st.subheader("🤖 AI")
        api_key = st.text_input("HuggingFace Token", type="password")
        if api_key:
            st.success("✅ AI ready")
        
        # Quick Info
        st.subheader("📈 Data Info")
        st.metric("Total Rows", f"{len(df):,}")
        if time_col and selected_years:
            st.metric("Years Selected", f"{min(selected_years)}-{max(selected_years)}")
    
    # Initialize engines
    engine = ComparisonEngine(df, time_col, metric_col)
    viz = VisualizationEngine()
    agent = AIAgent(api_key)
    
    # =========================================================================
    # MAIN TABS
    # =========================================================================
    tab_compare, tab_story = st.tabs(["🔬 Comparison Lab", "📚 Story Builder"])
    
    # =========================================================================
    # COMPARISON LAB
    # =========================================================================
    with tab_compare:
        analysis_cols = intel.get_analysis_columns()
        
        # ----- Configuration Section -----
        st.markdown("### ⚙️ Configure Comparison")
        
        col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1])
        
        with col1:
            primary_col = st.selectbox(
                "📊 Primary Dimension (X-axis)",
                analysis_cols,
                index=0,
                help="Main category to analyze"
            )
        
        with col2:
            remaining = [c for c in analysis_cols if c != primary_col]
            compare_col = st.selectbox(
                "🎨 Comparison Dimension (Color)",
                remaining,
                index=0,
                help="Secondary category for breakdown"
            )
        
        with col3:
            chart_type = st.selectbox(
                "📈 Chart Type",
                VisualizationEngine.CHART_TYPES,
                index=0
            )
        
        with col4:
            top_n = st.slider("Top N", 5, 25, 10)
        
        # ----- Filter Section -----
        with st.expander("🔍 **Filter Data** (Optional)", expanded=False):
            st.markdown("""
            <div class="explanation-box">
            <strong>How Filters Work:</strong><br>
            Filters restrict ALL data BEFORE analysis. For example, filtering by 
            <code>Funding Agency = "Department of Health"</code> means you ONLY see 
            records funded by that agency. This explains why some categories may have 
            limited values - that's what the filtered data actually contains.
            </div>
            """, unsafe_allow_html=True)
            
            filter_col = st.selectbox("Filter by Column", ['(No filter)'] + analysis_cols)
            
            filters = []
            if filter_col != '(No filter)':
                unique_vals = sorted(df[filter_col].astype(str).unique())
                
                if len(unique_vals) > 500:
                    st.warning(f"⚠️ {len(unique_vals)} unique values - showing first 200")
                    unique_vals = unique_vals[:200]
                
                filter_vals = st.multiselect(f"Select {filter_col} values", unique_vals)
                
                if filter_vals:
                    filters = [{'column': filter_col, 'values': filter_vals}]
                    
                    # Show filter impact
                    impact, filtered_df = engine.get_filter_impact(filters)
                    
                    st.markdown(f"""
                    <div class="filter-impact-box">
                    <strong>⚠️ Filter Impact:</strong><br>
                    Original: <strong>{impact['original']:,}</strong> rows → 
                    Filtered: <strong>{impact['filtered']:,}</strong> rows 
                    (<strong>{impact['pct_remaining']:.1f}%</strong> remaining)<br>
                    <em>Removed {impact['removed']:,} rows that don't match your filter.</em>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show what's in filtered data
                    st.caption(f"**In filtered data:** {compare_col} has {filtered_df[compare_col].nunique()} unique values")
        
        # ----- Explanation Box -----
        st.markdown(f"""
        <div class="data-context-box">
        <strong>📋 Current Analysis:</strong><br>
        Comparing <code>{primary_col}</code> broken down by <code>{compare_col}</code><br>
        Metric: <code>{metric_choice}</code> | Top: <code>{top_n}</code> entities | 
        Years: <code>{f'{min(selected_years)}-{max(selected_years)}' if selected_years else 'All'}</code>
        {f'<br>Filter: <code>{filter_col} = {", ".join(filter_vals[:3])}{"..." if len(filter_vals) > 3 else ""}</code>' if filters else ''}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ----- Generate Data -----
        if not time_col:
            st.warning("⚠️ Select a Time Column in sidebar for year-by-year analysis")
            st.stop()
        
        agg_df, top_entities, stats = engine.get_comparison_data(
            primary_col, compare_col, selected_years, filters, top_n
        )
        
        if agg_df.empty:
            st.error("❌ No data for current selection. Try different filters or parameters.")
            st.stop()
        
        # ----- Year-by-Year Charts -----
        st.markdown("### 📅 Year-by-Year Comparison")
        st.caption(f"Comparing **{primary_col}** × **{compare_col}** across {len(selected_years)} years")
        
        years_with_data = sorted(agg_df[time_col].unique())
        
        # Display in rows of 3
        for i in range(0, len(years_with_data), 3):
            batch = years_with_data[i:i+3]
            cols = st.columns(len(batch))
            
            for j, year in enumerate(batch):
                with cols[j]:
                    year_data = agg_df[agg_df[time_col] == year]
                    
                    st.markdown(f'<div class="year-badge">{year}</div>', unsafe_allow_html=True)
                    
                    fig = viz.create_chart(
                        year_data, primary_col, 'Value', compare_col,
                        chart_type if chart_type != 'Heatmap' else 'Grouped Bar',
                        title=""
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"year_{year}_{i}")
                    
                    # Year stats
                    year_total = year_data['Value'].sum()
                    is_currency = metric_col is not None
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{format_number(year_total, is_currency)}</div>
                        <div class="metric-label">Total</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ----- Advanced Analysis Section -----
        st.markdown("### 🔍 Advanced Diagnostics")
        
        tab_trend, tab_corr, tab_dist = st.tabs(["📈 Trend Analysis", "🔗 Correlation", "📊 Distribution"])
        
        with tab_trend:
            col_t1, col_t2 = st.columns([2, 1])
            with col_t1:
                trend_fig = viz.create_trend_chart(agg_df, time_col, primary_col)
                st.plotly_chart(trend_fig, use_container_width=True)
            with col_t2:
                st.markdown(f"""
                <div class="explanation-box">
                <strong>Trend Insight:</strong><br>
                Showing how top {primary_col} entities change over {time_col}.
                If lines are flat, check if data spans multiple periods.
                </div>
                """, unsafe_allow_html=True)

        with tab_corr:
            st.caption("Correlation between numeric variables (Pearson). Strong color = Strong relationship.")
            corr_fig = viz.create_correlation_matrix(df[numeric_cols] if numeric_cols else df)
            st.plotly_chart(corr_fig, use_container_width=True)
            
        with tab_dist:
            dist_fig = viz.create_distribution_plot(df, primary_col, metric_col)
            st.plotly_chart(dist_fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🗺️ Relationship Heatmap")
        summary_fig = viz.create_summary_heatmap(agg_df, primary_col, compare_col)
        st.plotly_chart(summary_fig, use_container_width=True)
        
        st.markdown("---")
        
        # ----- Findings -----
        st.markdown("### 📋 Key Findings")
        
        findings = generate_findings(
            stats, primary_col, compare_col, metric_choice,
            time_col, selected_years, agg_df
        )
        
        for finding in findings:
            st.markdown(f'<div class="finding-card">{finding}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ----- AI Analysis -----
        st.markdown("### 🤖 AI Analysis")
        
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_ai = st.button("🔬 Run AI Analysis", type="primary", use_container_width=True)
        with col_info:
            st.caption("AI analyzes the actual data and generates insights")
        
        if run_ai:
            with st.spinner("Analyzing..."):
                data_summary = generate_data_summary(
                    agg_df, primary_col, compare_col, time_col, stats, selected_years, filters
                )
                
                config = {
                    'col1': primary_col,
                    'col2': compare_col,
                    'years': f"{min(selected_years)}-{max(selected_years)}" if selected_years else "All",
                    'metric': metric_choice,
                    'filters': f"{filter_col}={filter_vals}" if filters else "None"
                }
                
                analysis = agent.analyze(data_summary, config)
                
                st.session_state['current_analysis'] = {
                    'text': analysis,
                    'summary': data_summary,
                    'findings': findings,
                    'config': config,
                    'stats': stats
                }
        
        # Display analysis
        if st.session_state.get('current_analysis'):
            analysis = st.session_state['current_analysis']
            
            st.markdown(analysis['text'])
            
            # Research option
            st.markdown("#### 🔍 Research a Finding")
            selected_finding = st.selectbox(
                "Select finding",
                analysis['findings'],
                format_func=lambda x: x[:80] + "..." if len(x) > 80 else x
            )
            
            if st.button("🌐 Search Web for Explanation"):
                with st.spinner("Searching..."):
                    # Generate search query
                    query = selected_finding.replace("**", "")[:100]
                    sources, context = agent.search_web(query)
                    
                    if sources:
                        explanation = agent.explain_finding(selected_finding, context)
                        st.markdown("**Explanation:**")
                        st.markdown(explanation)
                        
                        st.markdown("**Sources:**")
                        for s in sources:
                            st.markdown(f"- [{s['title'][:60]}]({s['url']})")
                    else:
                        st.info("No web sources found")
            
            # Save to story
            st.markdown("---")
            with st.expander("💾 Save to Story"):
                story_title = st.text_input("Title", f"{primary_col} vs {compare_col}")
                
                if st.button("➕ Add to Story"):
                    st.session_state['story_sections'].append({
                        'title': story_title,
                        'findings': analysis['findings'],
                        'analysis': analysis['text'],
                        'config': analysis['config'],
                        'timestamp': datetime.now().strftime("%H:%M")
                    })
                    st.success("✅ Added to story!")
    
    # =========================================================================
    # STORY BUILDER
    # =========================================================================
    with tab_story:
        st.markdown("### 📚 Your Data Story")
        
        if not st.session_state['story_sections']:
            st.info("👈 Add analysis sections from the Comparison Lab to build your story")
        else:
            st.success(f"📝 {len(st.session_state['story_sections'])} sections")
            
            for i, section in enumerate(st.session_state['story_sections']):
                with st.expander(f"📖 {section['title']}", expanded=True):
                    st.caption(f"Added: {section['timestamp']} | {section['config']}")
                    
                    st.markdown("**Findings:**")
                    for f in section['findings']:
                        st.markdown(f"- {f}")
                    
                    with st.expander("Full Analysis"):
                        st.markdown(section.get('analysis', ''))
                    
                    if st.button("🗑️ Remove", key=f"rm_{i}"):
                        st.session_state['story_sections'].pop(i)
                        st.rerun()
            
            st.markdown("---")
            
            if st.button("✨ Generate Combined Story", type="primary"):
                with st.spinner("Creating narrative..."):
                    story = agent.combine_story(st.session_state['story_sections'])
                    st.session_state['combined_story'] = story
            
            if st.session_state.get('combined_story'):
                st.markdown("### 📜 Final Story")
                st.markdown(st.session_state['combined_story'])
                
                c1, c2 = st.columns(2)
                c1.download_button("📥 Download MD", st.session_state['combined_story'], "story.md")
                c2.download_button("📥 Download TXT", st.session_state['combined_story'], "story.txt")
            
            if st.button("🗑️ Clear All"):
                st.session_state['story_sections'] = []
                st.session_state['combined_story'] = ""
                st.rerun()

if __name__ == "__main__":
    main()