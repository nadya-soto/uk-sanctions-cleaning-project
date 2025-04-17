# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

st.set_page_config(page_title="Sanctions Explorer", layout="wide")

st.title("UK Sanctions List Explorer 1")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/ethel/uk-sanctions-cleaning-project/uk-sanctions-cleaning-project/output/UK_Sanctions_Processed.csv", parse_dates=["listed_on", "uk_sanctions_date", "last_updated", "DOB"], dayfirst=True)
    return df

df = load_data()

st.sidebar.header("Filters")
entity_type = st.sidebar.multiselect("Entity type", df["group_type"].unique(), default=df["group_type"].unique())
regime = st.sidebar.multiselect("Regime", df["regime"].unique(), default=df["regime"].unique())

filtered_df = df[df["group_type"].isin(entity_type) & df["regime"].isin(regime)]

# Alias count
filtered_df["alias_count"] = filtered_df["aliases"].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)


# Data Quality Dashboard
st.header("Data Quality Dashboard", divider="rainbow")

#  Summary Cards
st.subheader("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", len(filtered_df), help="All entries in current filtered view")
with col2:
    st.metric("Unique Entities", filtered_df["primary_name"].nunique(), 
             delta=f"{(1-filtered_df['primary_name'].duplicated().mean())*100:.1f}% unique")
with col3:
    st.metric("Complete Records", 
             len(filtered_df.dropna(subset=["DOB", "addresses", "nationality"])),
             help="Records with all key fields populated")
with col4:
    completeness_score = filtered_df.notna().mean().mean()
    st.metric("Overall Completeness", f"{completeness_score*100:.1f}%")

# Interactive Missing Data Heatmap
st.subheader("Missing Data Heatmap")
missing_pct = filtered_df.isna().mean().sort_values(ascending=False)
fig = px.imshow(filtered_df.isna().astype(int).T,
               labels=dict(x="Record", y="Field", color="Missing"),
               color_continuous_scale='Reds',
               title="Field Completeness Across Records")
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# Field Completeness Breakdown
st.subheader("Field Completeness")
tab1, tab2 = st.tabs(["By Field", "By Entity Type"])

with tab1:
    field_stats = pd.DataFrame({
        "Field": missing_pct.index,
        "Missing %": (missing_pct*100).round(1),
        "Complete Count": (len(filtered_df)*(1-missing_pct)).astype(int)
    })
    
    fig = px.bar(field_stats.head(10), 
                x="Field", y="Missing %",
                hover_data=["Complete Count"],
                color="Missing %",
                color_continuous_scale="RdYlGn_r",
                title="Top Incomplete Fields")
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View all field stats"):
        st.dataframe(field_stats.style.background_gradient(cmap="Reds", subset=["Missing %"]),
                    use_container_width=True)

with tab2:
    entity_quality = filtered_df.groupby('group_type').agg({
        'primary_name': 'count',
        'DOB': lambda x: x.notna().mean()*100,
        'addresses': lambda x: x.notna().mean()*100,
        'nationality': lambda x: x.notna().mean()*100
    }).reset_index()
    
    entity_quality = entity_quality.rename(columns={
        'primary_name': 'Count',
        'DOB': 'DOB % Complete',
        'addresses': 'Address % Complete',
        'nationality': 'Nationality % Complete'
    })
    
    st.plotly_chart(px.bar(entity_quality, 
                          x='group_type', y=['DOB % Complete', 'Address % Complete', 'Nationality % Complete'],
                          barmode='group',
                          title="Completeness by Entity Type"),
                   use_container_width=True)

# Advanced Duplicate Analysis
st.subheader("Potential Duplicates Analysis")

if 'primary_name' in filtered_df.columns:
    dupes = filtered_df[filtered_df.duplicated('primary_name', keep=False)].sort_values('primary_name')
    
    if not dupes.empty:
        st.warning(f"Found {len(dupes)} potential duplicates by name")
        
        # Interactive selection
        selected_name = st.selectbox("Inspect potential duplicates:", 
                                    dupes['primary_name'].unique())
        
        name_group = dupes[dupes['primary_name'] == selected_name]
        
        cols = st.columns([1,3])
        with cols[0]:
            st.metric("Records", len(name_group))
            st.metric("Unique DOBs", name_group['DOB'].nunique())
            st.metric("Nationalities", name_group['nationality'].nunique())
        
        with cols[1]:
            st.dataframe(name_group[[
                'group_type', 'DOB', 'nationality', 
                'regime', 'listed_on'
            ]].style.highlight_null(color='red'),
            use_container_width=True)
            
            # Simple similarity comparison
            if len(name_group) > 1:
                st.write("**Field Matching:**")
                matches = {
                    'DOB': name_group['DOB'].nunique() == 1,
                    'Nationality': name_group['nationality'].nunique() == 1,
                    'Regime': name_group['regime'].nunique() == 1
                }
                for field, is_match in matches.items():
                    st.write(f"{field}: {'Yes' if is_match else 'No'}")
    else:
        st.success("No exact name duplicates found", icon="Yes")
else:
    st.error("Primary name field not available for duplicate check")

# Data Quality Over Time
st.subheader("Completeness Over Time")
if 'listed_on' in filtered_df.columns:
    time_completeness = filtered_df.set_index('listed_on').resample('Y').agg({
        'DOB': lambda x: x.notna().mean(),
        'addresses': lambda x: x.notna().mean(),
        'nationality': lambda x: x.notna().mean()
    })
    
    st.plotly_chart(px.line(time_completeness*100,
                           labels={'value': 'Completeness %'},
                           title="Field Completeness Over Time"),
                   use_container_width=True)

# Most frequent regimes
st.subheader("Top regimes")
st.plotly_chart(px.histogram(filtered_df, x="regime", color="regime", title="Frequency by regime"))

# Nationalities
st.subheader("Top nationalities")
nats = filtered_df["nationality"].dropna().str.strip("[]").str.replace("'", "").str.split(", ")
nats_exploded = nats.explode().value_counts().head(10)
st.bar_chart(nats_exploded)

# Sanction dates
st.subheader("Sanction timeline")
filtered_df["year_listed"] = filtered_df["listed_on"].dt.year
st.plotly_chart(px.histogram(filtered_df, x="year_listed", title="Number of sanctions per year"))

# Alias Analysis Center

st.header("Alias Analysis Center")
tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Frequency", "Patterns", "Lookup"])

with tab1:
    st.subheader("Alias Count Distribution")
    col1, col2 = st.columns([3,1])
    with col1:
        st.plotly_chart(px.histogram(
            filtered_df, 
            x="alias_count", 
            nbins=100,
            title="Number of Aliases per Entity",
            labels={"alias_count": "Number of Aliases"}
        ), use_container_width=True)
    with col2:
        st.metric("Max Aliases", filtered_df['alias_count'].max())
        st.metric("Avg Aliases", round(filtered_df['alias_count'].mean(), 1))
        st.metric("Single-Alias Entities", 
                len(filtered_df[filtered_df['alias_count'] == 1]))

with tab2:
    st.subheader("Alias Frequency Analysis")
    common_aliases = pd.Series(
        [alias for sublist in filtered_df['aliases'].dropna().apply(eval) 
         for alias in sublist]
    ).value_counts().head(30)

    fig = px.bar(
        common_aliases,
        orientation='h',
        labels={'index':'Alias', 'value':'Count'},
        title="Top 30 Most Reused Aliases"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Full Alias List",
        common_aliases.to_csv(),
        "alias_frequency.csv",
        help="CSV containing all aliases and their frequency counts"
    )

with tab3:
    st.subheader("Name Pattern Detection")

    # Component Analysis
    st.markdown("### Common Name Components")
    all_aliases = [alias for sublist in filtered_df['aliases'].dropna().apply(eval) 
                  for alias in sublist]
    components = pd.Series(
        [part for alias in all_aliases for part in alias.split()]
    ).value_counts().head(15)

    st.plotly_chart(px.bar(
        components,
        title="Most Frequent Name Parts",
        labels={'index':'Name Part', 'value':'Count'}
    ), use_container_width=True)

    # Similarity Finder
    st.markdown("### Find Similar Aliases")
    selected_alias = st.selectbox(
        "Select an alias to analyze:", 
        common_aliases.index.tolist()
    )

    similar = [a for a in all_aliases 
              if any(part in a for part in selected_alias.split())]
    similar_counts = pd.Series(similar).value_counts().head(20)

    if len(similar_counts) > 1:
        st.write(f"Aliases containing parts of '{selected_alias}':")
        st.dataframe(similar_counts)
    else:
        st.info("No similar aliases found beyond the selected one")

with tab4:
    st.subheader("Entity Lookup by Alias")

    search_col, filter_col = st.columns([3,1])
    with search_col:
        alias_search = st.text_input(
            "Search aliases:",
            placeholder="Enter full or partial alias"
        )
    with filter_col:
        min_aliases = st.number_input(
            "Min aliases", 
            min_value=1, 
            max_value=10, 
            value=1
        )

    if alias_search:
        matches = filtered_df[
            (filtered_df['aliases'].apply(
                lambda x: alias_search.lower() in str(x).lower() 
                if pd.notna(x) else False
            )) &
            (filtered_df['alias_count'] >= min_aliases)
        ]

        if not matches.empty:
            st.success(f"Found {len(matches)} matching entities")
            for _, row in matches.iterrows():
                with st.expander(f"{row['primary_name']} ({row['regime']})"):
                    cols = st.columns([2,1,1])
                    cols[0].write(f"**Aliases:** {', '.join(eval(row['aliases']))}")
                    cols[1].write(f"**Listed:** {row['listed_on'].strftime('%Y-%m-%d')}")
                    cols[2].write(f"**Alias Count:** {row['alias_count']}")
        else:
            st.warning("No entities found matching this alias")
'''
# NLP Section
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]   

st.header("NLP Insights")

#Relations etwee entities
st.subheader("Knowledge Graph: Relationships between Entities")

# 100 text
sampled_infos = filtered_df['other_information'].dropna().sample(n=100, random_state=42)
docs = list(nlp.pipe(sampled_infos, batch_size=20))

# Graph
G = nx.Graph()
for doc in docs:
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            G.add_edge(entities[i][0], entities[j][0])

# Limit
if len(G.nodes) > 100:
    G = G.subgraph(list(G.nodes)[:100])

plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1500, font_size=10)
st.pyplot(plt)

# Word Cloud
st.subheader("Word Cloud from Other Information")
text_data = " ".join(filtered_df['other_information'].dropna().tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
st.image(wordcloud.to_array(), use_column_width=True)

# Frequency Analysis
st.subheader("Frequent Terms in Descriptions")
vectorizer = CountVectorizer(stop_words='english', max_features=30)
X = vectorizer.fit_transform(filtered_df['other_information'].dropna())
frequencies = pd.Series(X.toarray().sum(axis=0), index=vectorizer.get_feature_names_out())
st.bar_chart(frequencies.sort_values(ascending=False))

# Sample Entity Extraction
st.subheader("Sample Named Entity Recognition Examples")
sample_texts = filtered_df["other_information"].dropna().sample(5, random_state=42).tolist()
for idx, text in enumerate(sample_texts):
    doc = nlp(text)
    st.markdown(f"### Example {idx+1}")
    st.write(f"**Original text:** {text}")
    if doc.ents:
        ents_data = [(ent.text, ent.label_) for ent in doc.ents]
        ents_df = pd.DataFrame(ents_data, columns=["Entity", "Label"])
        st.dataframe(ents_df)
    else:
        st.info("No named entities found in this sample.")
        

'''