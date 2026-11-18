from pathlib import Path

import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go

pn.extension("plotly")

BASE_DIR = Path(__file__).resolve().parent

df_by_state_and_oc = pd.read_csv(
    BASE_DIR / "state_occupation_summary.csv",
    index_col=["PRIM_STATE", "OCC_TITLE"],
)

df_by_occupation = pd.read_csv(
    BASE_DIR / "occupation_summary.csv",
    index_col="OCC_TITLE",
)


def make_state_comparison(state_1, state_2, statistic="H_MEDIAN"):
    """Compare two states using a dot plot of occupational summaries."""

    if state_1 == state_2:
        raise ValueError("Select two different states.")

    # Both state and occupation are required for this chart.
    data = df_by_state_and_oc.copy()


    data = data.reset_index()

    # Summary Statistic = all non-grouped columns except JOBS_1000.
    valid_statistics = [
        col for col in data.columns
        if col not in ["PRIM_STATE", "OCC_TITLE", "JOBS_1000"]
    ]

    if statistic not in valid_statistics:
        raise ValueError(
            f"Choose a Summary Statistic from: {valid_statistics}"
        )

    data[statistic] = pd.to_numeric(
        data[statistic], errors="coerce"
    )

    selected = data.loc[
        data["PRIM_STATE"].isin([state_1, state_2]),
        ["PRIM_STATE", "OCC_TITLE", statistic],
    ].dropna(subset=["OCC_TITLE", statistic])

    # Keep occupations with valid values in BOTH states.
    paired = selected.pivot(
        index="OCC_TITLE",
        columns="PRIM_STATE",
        values=statistic,
    )

    paired = paired.reindex(
        columns=[state_1, state_2]
    ).dropna(subset=[state_1, state_2])

    # Sort occupations by their average value across the two states.
    occupation_order = (
        paired.mean(axis=1)
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Return to long format for Plotly.
    plot_data = (
        paired.reset_index()
        .melt(
            id_vars="OCC_TITLE",
            value_vars=[state_1, state_2],
            var_name="PRIM_STATE",
            value_name="VALUE",
        )
    )

    units = " ($/hour)" if statistic.startswith("H_") else ""

    fig = px.scatter(
        plot_data,
        x="VALUE",
        y="OCC_TITLE",
        color="PRIM_STATE",
        symbol="PRIM_STATE",
        color_discrete_map={
            state_1: "#0072B2",
            state_2: "#D55E00",
        },
        symbol_map={
            state_1: "circle",
            state_2: "square",
        },
        category_orders={
            "PRIM_STATE": [state_1, state_2],
            "OCC_TITLE": occupation_order,
        },
        labels={
            "VALUE": f"{statistic}{units}",
            "OCC_TITLE": "Occupational group",
            "PRIM_STATE": "State",
        },
        title=f"{statistic} by occupational group: {state_1} versus {state_2}",
    )

    fig.update_traces(
        marker=dict(size=11, opacity=0.85),
    )

    fig.update_layout(
        template="plotly_white",
        height=max(450, len(occupation_order) * 30 + 150),
        margin=dict(l=30, r=30, t=80, b=60),
        legend_title_text="State",
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True, automargin=True)

    if paired.empty:
        fig.add_annotation(
            text="No occupational groups have valid values in both states.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    return fig
# Preview

def make_occupation_boxplot(selected_occupations):
    """Create percentile boxes without whiskers for up to five groups."""

    if selected_occupations is None:
        selected_occupations = []

    # Accept either one occupation string or a list.
    if isinstance(selected_occupations, str):
        selected_occupations = [selected_occupations]

    # Remove duplicate selections while preserving their order.
    selected_occupations = list(
        dict.fromkeys(selected_occupations)
    )

    if len(selected_occupations) > 5:
        raise ValueError(
            "Select no more than five occupational groups."
        )

    data = df_by_occupation.copy()


    data = data.reset_index()

    percentile_columns = [
    "H_PCT10",
    "H_PCT25",
    "H_MEDIAN",
    "H_PCT75",
    "H_PCT90",
    ]

    data[percentile_columns] = data[percentile_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    available = set(data["OCC_TITLE"].dropna())

    unknown = [
        occupation
        for occupation in selected_occupations
        if occupation not in available
    ]

    if unknown:
        raise ValueError(
            f"Unknown occupational groups: {unknown}"
        )

    selected = (
        data.set_index("OCC_TITLE")
        .reindex(selected_occupations)
        .dropna(subset=percentile_columns)
    )

    # Check that the median falls inside the box.
    ordered = (
        (selected["H_PCT10"] <= selected["H_PCT25"])
        & (selected["H_PCT25"] <= selected["H_MEDIAN"])
        & (selected["H_MEDIAN"] <= selected["H_PCT75"])
        & (selected["H_PCT75"] <= selected["H_PCT90"])
    )

    if not ordered.all():
        invalid_groups = selected.index[~ordered].tolist()
        raise ValueError(
            f"Percentiles are out of order for: {invalid_groups}"
        )

    fig = go.Figure()

    for occupation, row in selected.iterrows():
        fig.add_trace(
            go.Box(
                x=[occupation],
                name=occupation,
                q1=[row["H_PCT25"]],
                median=[row["H_MEDIAN"]],
                q3=[row["H_PCT75"]],

                # Whiskers at the 10th and 90th percentiles.
                lowerfence=[row["H_PCT10"]],
                upperfence=[row["H_PCT90"]],
                whiskerwidth=0.5,

                boxpoints=False,
                showlegend=False,
                fillcolor="rgba(0, 114, 178, 0.35)",
                line=dict(color="#0072B2", width=2),
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                "Occupational wage percentile summaries"
                "<br><sup>"
                "Box: 25th–75th percentiles | Middle line: median | "
                "**Non-Standard Whiskers: 10th and 90th percentiles"
                "</sup>"
            )
        ),
        xaxis_title="Occupational group",
        yaxis_title="Hourly wage summary ($/hour)",
        boxmode="group",
        template="plotly_white",
        height=600,
        margin=dict(l=60, r=30, t=110, b=160),
    )

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=selected_occupations,
        tickangle=-25,
        automargin=True,
    )

    fig.update_yaxes(
        showgrid=True,
        rangemode="tozero",
    )

    if selected.empty:
        fig.add_annotation(
            text="Select up to five occupational groups with valid data.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    return fig


def make_occupation_choropleth(occupation):
    """Map jobs per 1,000 jobs by state for one occupational group."""

    # PRIM_STATE and OCC_TITLE are index levels.
    data = df_by_state_and_oc.copy().reset_index()

    if occupation not in data["OCC_TITLE"].values:
        raise ValueError(
            f"Unknown occupational group: {occupation}"
        )

    selected = data.loc[
        data["OCC_TITLE"] == occupation,
        ["PRIM_STATE", "JOBS_1000"],
    ].copy()

    selected["JOBS_1000"] = pd.to_numeric(
        selected["JOBS_1000"],
        errors="coerce",
    )

    selected["PRIM_STATE"] = (
        selected["PRIM_STATE"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if selected["PRIM_STATE"].duplicated().any():
        raise ValueError(
            "Expected one row per state for the selected occupation."
        )

    # Include all 50 states and DC, even when estimates are missing.
    state_codes = (
        "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY "
        "LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH "
        "OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY"
    ).split()

    map_data = pd.DataFrame(
        {"PRIM_STATE": state_codes}
    ).merge(
        selected,
        on="PRIM_STATE",
        how="left",
        validate="one_to_one",
    )

    fig = px.choropleth(
        map_data,
        locations="PRIM_STATE",
        locationmode="USA-states",
        color="JOBS_1000",
        scope="usa",
        color_continuous_scale="Viridis",
        labels={
            "PRIM_STATE": "State",
            "JOBS_1000": "Jobs per 1,000 jobs",
        },
    )

    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.6,
        hovertemplate=(
            "<b>%{location}</b><br>"
            + f"{occupation}<br>"
            + "Jobs per 1,000 jobs: %{z:,.2f}"
            + "<extra></extra>"
        ),
    )

    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="#E5E5E5",
        showsubunits=True,
        subunitcolor="white",
        showlakes=False,
        showframe=False,
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Occupational employment concentration: {occupation}"
                "<br><sup>"
                "Jobs per 1,000 jobs by state. "
                "Uncolored states have no valid estimate."
                "</sup>"
            )
        ),
        template="plotly_white",
        height=600,
        margin=dict(l=10, r=10, t=100, b=20),
        coloraxis_colorbar=dict(
            title="Jobs per<br>1,000 jobs",
            tickformat=",.2f",
        ),
    )

    if map_data["JOBS_1000"].notna().sum() == 0:
        fig.add_annotation(
            text="No valid estimates for this occupational group.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    return fig

def make_occupation_bullet(
    occupation,
    state,
    statistic="H_MEDIAN",
):
    """Compare a state estimate with its national occupational summary."""

    state_data = df_by_state_and_oc.copy()
    national_data = df_by_occupation.copy()

    # Summary Statistic = all non-grouped columns except JOBS_1000.
    valid_statistics = [
        column
        for column in state_data.columns
        if column != "JOBS_1000"
        and column in national_data.columns
    ]

    if statistic not in valid_statistics:
        raise ValueError(
            f"Choose a Summary Statistic from: {valid_statistics}"
        )

    # Both DataFrames retain their grouping indexes.
    state_key = (state, occupation)

    if state_key not in state_data.index:
        raise ValueError(
            f"No state summary found for {occupation} in {state}."
        )

    if occupation not in national_data.index:
        raise ValueError(
            f"No national summary found for {occupation}."
        )

    state_value = pd.to_numeric(
        state_data.loc[state_key, statistic],
        errors="coerce",
    )

    national_value = pd.to_numeric(
        national_data.loc[occupation, statistic],
        errors="coerce",
    )

    if pd.isna(state_value) or pd.isna(national_value):
        raise ValueError(
            "The selected state or national summary is missing "
            "or nonnumeric."
        )

    state_value = float(state_value)
    national_value = float(national_value)

    units = " ($/hour)" if statistic.startswith("H_") else ""
    prefix = "$" if statistic.startswith("H_") else ""

    # Leave space beyond both the state bar and reference line.
    axis_max = max(state_value, national_value, 1.0) * 1.2

    fig = go.Figure(
        go.Indicator(
            mode="number+gauge",
            value=state_value,
            number=dict(
                prefix=prefix,
                valueformat=",.2f",
            ),
            gauge=dict(
                shape="bullet",
                axis=dict(
                    range=[0, axis_max],
                    tickformat=",.2f",
                    tickprefix=prefix,
                ),
                bar=dict(
                    color="#0072B2",
                    thickness=0.5,
                ),
                bgcolor="#EEEEEE",

                # Reference line is the selected NATIONAL statistic.
                threshold=dict(
                    value=national_value,
                    line=dict(
                        color="#222222",
                        width=4,
                    ),
                    thickness=0.85,
                ),
            ),
            domain=dict(
                x=[0.05, 0.85],
                y=[0.25, 0.75],
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{occupation}: {state} versus national summary"
                f"<br><sup>{statistic}{units}</sup>"
            )
        ),
        template="plotly_white",
        height=350,
        margin=dict(l=50, r=50, t=100, b=80),
    )

    fig.add_annotation(
        text=(
            f"Bar — {state}: {prefix}{state_value:,.2f}"
            f" | Reference line — national summary: "
            f"{prefix}{national_value:,.2f}"
        ),
        x=0.5,
        y=0,
        xref="paper",
        yref="paper",
        showarrow=False,
    )

    return fig


state_occupation_data = df_by_state_and_oc.reset_index()

states = sorted(
    state_occupation_data["PRIM_STATE"]
    .dropna()
    .unique()
    .tolist()
)

# Use occupations available in both DataFrames.
occupations = sorted(
    set(state_occupation_data["OCC_TITLE"].dropna())
    & set(df_by_occupation.index)
)

# Summary Statistic = non-grouped columns except JOBS_1000.
# The bullet chart also requires the statistic in df_by_occupation.
summary_statistics = [
    column
    for column in df_by_state_and_oc.columns
    if column not in ["PRIM_STATE", "OCC_TITLE", "JOBS_1000"]
    and column in df_by_occupation.columns
]

if len(states) < 2:
    raise ValueError("The dashboard requires at least two states.")

if not occupations:
    raise ValueError("No occupational groups are available in both DataFrames.")

if not summary_statistics:
    raise ValueError("No shared Summary Statistic columns are available.")

default_statistic = (
    "H_MEDIAN"
    if "H_MEDIAN" in summary_statistics
    else summary_statistics[0]
)

default_occupation = (
    "Management"
    if "Management" in occupations
    else occupations[0]
)

default_state_1 = "MI" if "MI" in states else states[0]

default_state_2 = (
    "CA"
    if "CA" in states and default_state_1 != "CA"
    else next(state for state in states if state != default_state_1)
)


# --------------------------------------------------
# 2. Shared widgets
# --------------------------------------------------

summary_stat_widget = pn.widgets.Select(
    name="Summary Statistic — dot plot and bullet chart",
    options=summary_statistics,
    value=default_statistic,
    width=350,
)

occupation_widget = pn.widgets.Select(
    name="Occupational group — map and bullet chart",
    options=occupations,
    value=default_occupation,
    width=400,
)


# --------------------------------------------------
# 3. Chart-specific widgets
# --------------------------------------------------

# Dot plot controls.
state_1_widget = pn.widgets.Select(
    name="State 1",
    options=states,
    value=default_state_1,
    width=200,
)

state_2_widget = pn.widgets.Select(
    name="State 2",
    options=states,
    value=default_state_2,
    width=200,
)

# Boxplot control.
box_occupations_widget = pn.widgets.MultiChoice(
    name="Occupational groups — select up to 5",
    options=occupations,
    value=occupations[:3],
    sizing_mode="stretch_width",
)

# Bullet chart control.
bullet_state_widget = pn.widgets.Select(
    name="State",
    options=states,
    value=default_state_1,
    width=200,
)


# --------------------------------------------------
# 4. Rendering helpers
# --------------------------------------------------

def plotly_pane(figure):
    return pn.pane.Plotly(
        figure,
        sizing_mode="stretch_width",
        config={
            "displaylogo": False,
            "responsive": True,
        },
    )


def render_dotplot(state_1, state_2, statistic):
    if state_1 == state_2:
        return pn.pane.Markdown(
            "**Select two different states to compare.**"
        )

    try:
        return plotly_pane(
            make_state_comparison(
                state_1,
                state_2,
                statistic,
            )
        )
    except ValueError as error:
        return pn.pane.Markdown(f"**Chart unavailable:** {error}")


def render_boxplot(selected_occupations):
    if not selected_occupations:
        return pn.pane.Markdown(
            "**Select between one and five occupational groups.**"
        )

    if len(selected_occupations) > 5:
        return pn.pane.Markdown(
            "**Select no more than five occupational groups.** "
            "Remove a selection to display the chart."
        )

    try:
        return plotly_pane(
            make_occupation_boxplot(selected_occupations)
        )
    except ValueError as error:
        return pn.pane.Markdown(f"**Chart unavailable:** {error}")


def render_map(occupation):
    try:
        return plotly_pane(
            make_occupation_choropleth(occupation)
        )
    except ValueError as error:
        return pn.pane.Markdown(f"**Chart unavailable:** {error}")


def render_bullet(occupation, state, statistic):
    try:
        return plotly_pane(
            make_occupation_bullet(
                occupation,
                state,
                statistic,
            )
        )
    except ValueError as error:
        return pn.pane.Markdown(f"**Chart unavailable:** {error}")


# --------------------------------------------------
# 5. Bind charts to their widgets
# --------------------------------------------------

dotplot_view = pn.bind(
    render_dotplot,
    state_1=state_1_widget,
    state_2=state_2_widget,
    statistic=summary_stat_widget,
)

boxplot_view = pn.bind(
    render_boxplot,
    selected_occupations=box_occupations_widget,
)

map_view = pn.bind(
    render_map,
    occupation=occupation_widget,
)

bullet_view = pn.bind(
    render_bullet,
    occupation=occupation_widget,
    state=bullet_state_widget,
    statistic=summary_stat_widget,
)


# --------------------------------------------------
# 6. Create chart sections
# --------------------------------------------------

dotplot_section = pn.Column(
    "## 1. Occupational earnings comparison",
    "Compare all occupational groups between two states.",
    pn.Row(state_1_widget, state_2_widget),
    dotplot_view,
    sizing_mode="stretch_width",
)

boxplot_section = pn.Column(
    "## 2. Occupational wage percentile summaries",
    box_occupations_widget,
    boxplot_view,
    (
        "**Box:** 25th–75th percentiles. "
        "**Middle line:** median. "
        "**Whiskers:** 10th and 90th percentiles."
    ),
    sizing_mode="stretch_width",
)

map_section = pn.Column(
    "## 3. Occupational employment concentration",
    (
        "The shared occupational-group dropdown controls this map. "
        "Colors represent jobs per 1,000 jobs—not job vacancies."
    ),
    map_view,
    sizing_mode="stretch_width",
)

bullet_section = pn.Column(
    "## 4. State versus national occupational summary",
    bullet_state_widget,
    bullet_view,
    (
        "**Bar:** selected state's value. "
        "**Reference line:** the same statistic and occupation "
        "in `df_by_occupation`."
    ),
    sizing_mode="stretch_width",
)


# --------------------------------------------------
# 7. Assemble the dashboard on one page
# --------------------------------------------------

dashboard = pn.Column(
    "# U.S. Occupational Employment and Wage Dashboard",

    "## Shared controls",
    pn.Row(
        summary_stat_widget,
        occupation_widget,
    ),
    (
        "The Summary Statistic controls Charts 1 and 4. "
        "The single occupational group controls Charts 3 and 4. "
        "Chart 2 has its own multiple-group selector."
    ),

    pn.layout.Divider(),
    dotplot_section,

    pn.layout.Divider(),
    boxplot_section,

    pn.layout.Divider(),
    map_section,

    pn.layout.Divider(),
    bullet_section,

    sizing_mode="stretch_width",
)
# Keep the dashboard responsive, but prevent excessive stretching.
dashboard.sizing_mode = "stretch_width"
dashboard.max_width = 1400
dashboard.margin = (15, 20)

dashboard.servable(
    title="U.S. Occupational Employment and Wage Dashboard"
)
