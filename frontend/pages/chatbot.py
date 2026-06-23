"""
pages/chatbot.py — Interactive RAG Chatbot Interface.
Connects to the FastAPI backend which routes to the local Ollama RAGAgent.
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import ask_chatbot

dash.register_page(__name__, path="/chatbot", name="Assistant IA", title="Assistant IA")

def layout():
    return dbc.Container([
        html.H2("🤖 Assistant IA — OCP Cogénération", className="text-light my-4"),
        
        # Settings & Controls
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Fenêtre de Contexte (Heures)", className="text-light small"),
                        dcc.Slider(
                            id="chat-context-hours",
                            min=1, max=168, step=1, value=24,
                            marks={1: "1h", 24: "24h", 72: "3j", 168: "7j"},
                            className="text-light"
                        ),
                    ], width=8),
                    dbc.Col([
                        html.Label("Actions", className="text-light small d-block"),
                        dbc.Button(
                            "🗑️ Effacer l'historique", 
                            id="clear-chat-btn", 
                            color="danger", 
                            size="sm", 
                            className="mt-1 w-100"
                        ),
                    ], width=4),
                ])
            ])
        ], className="bg-dark border-secondary mb-3"),

        # Chat Window with Loading Spinner
        dbc.Card([
            dbc.CardBody([
                dcc.Loading(
                    id="loading-chat",
                    type="circle",
                    color="#f0c040", # OCP Yellow spinner
                    children=html.Div(
                        id="chat-window",
                        style={
                            "height": "55vh",
                            "overflowY": "auto",
                            "padding": "15px",
                            "backgroundColor": "#121212",
                            "borderRadius": "8px",
                            "border": "1px solid #333"
                        },
                        children=[
                            html.Div(
                                "Bienvenue ! Posez-moi une question sur la centrale (ex: 'Pourquoi le GTA2 a-t-il un faible rendement ?')", 
                                className="text-muted text-center my-5"
                            )
                        ]
                    )
                )
            ])
        ], className="bg-dark border-secondary mb-3"),

        # Input Area
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Input(
                            id="chat-input",
                            type="text",
                            placeholder="Posez votre question ici... (Appuyez sur Entrée pour envoyer)",
                            className="bg-dark text-light border-secondary",
                            style={"height": "45px"},
                            autoComplete="off"
                        )
                    ], width=10),
                    dbc.Col([
                        dbc.Button(
                            "Envoyer 🚀", 
                            id="send-btn", 
                            color="warning", 
                            className="w-100",
                            style={"height": "45px"}
                        )
                    ], width=2),
                ])
            ])
        ], className="bg-dark border-secondary"),
        
        dcc.Store(id="chat-history-store", data=[]),
    ], fluid=True, className="p-4")

def render_chat_window(history):
    """Renders the chat bubbles with Markdown support."""
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(
                html.Div([
                    html.Div("Vous", className="fw-bold text-warning small", style={"textAlign": "right"}),
                    html.Div(msg["content"], className="text-light mt-1", style={"whiteSpace": "pre-wrap", "textAlign": "right"})
                ], className="mb-3 p-3 rounded", style={"backgroundColor": "#2a2a3e", "marginLeft": "20%"})
            )
        elif msg["role"] == "assistant":
            # Build sources and confidence badges
            badges = []
            if msg.get("confidence"):
                badges.append(html.Span(f"Confiance: {msg['confidence']*100:.0f}%", className="badge bg-success ms-2"))
            if msg.get("model"):
                badges.append(html.Span(f"Modèle: {msg['model']}", className="badge bg-secondary ms-2"))
                
            sources_badge = html.Div()
            if msg.get("sources"):
                sources_str = ", ".join(msg["sources"])
                sources_badge = html.Div([
                    html.Small("Sources: ", className="text-muted"),
                    html.Small(sources_str, className="text-info")
                ], className="mt-2")

            messages.append(
                html.Div([
                    html.Div([
                        html.Span("Assistant IA 🤖", className="fw-bold text-success small"),
                        *badges
                    ]),
                    # Use dcc.Markdown for beautiful formatting of the AI's technical answers
                    dcc.Markdown(msg["content"], className="text-light mt-2"),
                    sources_badge
                ], className="mb-3 p-3 rounded", style={"backgroundColor": "#1e1e2e", "marginRight": "20%"})
            )
            
    return messages

@callback(
    Output("chat-history-store", "data"),
    Output("chat-window", "children"),
    Output("chat-input", "value"),
    Input("send-btn", "n_clicks"),
    Input("chat-input", "n_submit"), # Allows sending by pressing Enter
    Input("clear-chat-btn", "n_clicks"),
    State("chat-input", "value"),
    State("chat-history-store", "data"),
    State("chat-context-hours", "value"),
    prevent_initial_call=True
)
def handle_chat_interaction(send_clicks, n_submit, clear_clicks, user_input, history, context_hours):
    triggered_id = ctx.triggered_id
    
    # Handle Clear History
    if triggered_id == "clear-chat-btn":
        empty_state = [html.Div("Historique effacé. Posez une nouvelle question !", 
                                className="text-muted text-center my-5")]
        return [], empty_state, ""
        
    # Handle Send Message
    if triggered_id in ["send-btn", "chat-input"] and user_input:
        if history is None:
            history = []
            
        # 1. Add user message to history
        history.append({"role": "user", "content": user_input})
        
        # 2. Call backend API (This will trigger the Ollama generation)
        response = ask_chatbot(user_input, context_hours=context_hours)
        
        # 3. Extract data safely
        ai_text = response.get("response", "Désolé, je n'ai pas pu générer de réponse.")
        sources = response.get("sources", [])
        confidence = response.get("confidence", 0)
        model = response.get("model", "local")
        
        # 4. Add AI message to history
        history.append({
            "role": "assistant",
            "content": ai_text,
            "sources": sources,
            "confidence": confidence,
            "model": model
        })
        
        # 5. Render chat window
        chat_window = render_chat_window(history)
        
        return history, chat_window, ""
        
    raise dash.exceptions.PreventUpdate