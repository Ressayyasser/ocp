COLORS = {
    'bg_dark': '#0d1117',
    'bg_card': '#161b22',
    'bg_panel': '#21262d',
    'accent_blue': '#1f6feb',
    'accent_orange': '#f0883e',
    'accent_green': '#3fb950',
    'accent_red': '#f85149',
    'accent_yellow': '#d29922',
    'accent_cyan': '#39c5cf',
    'text_primary': '#e6edf3',
    'text_muted': '#8b949e',
    'border': '#30363d',
}

card_style = {
    'backgroundColor': COLORS['bg_card'],
    'border': f'1px solid {COLORS["border"]}',
    'borderRadius': '8px',
    'padding': '16px',
    'marginBottom': '16px',
}

kpi_card_style = {
    **card_style,
    'textAlign': 'center',
    'borderTop': f'3px solid {COLORS["accent_blue"]}',
}