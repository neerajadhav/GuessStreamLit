import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import time
import random
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()

# Constants and Configuration
class GameConfig:
    # API and Quota Management
    MAX_API_CALLS_PER_SESSION = 20
    DEFAULT_MODEL = 'gemini-2.5-flash'
    
    # Game Settings
    DEFAULT_ROUND_OPTIONS = [3, 5, 7]
    DIFFICULTY_OPTIONS = ["easy", "medium", "hard"]
    DEFAULT_DIFFICULTY = "medium"
    MIN_PLAYERS = 2
    MAX_PLAYERS = 5
    
    # Scoring System
    SCORE_PERFECT = 100
    SCORE_CLOSE = 80
    SCORE_RELATED = 60
    SCORE_DISTANT = 20
    SCORE_MIN = 0
    SCORE_MAX = 100
    
    # Model Categories for Filtering
    SUPPORTED_MODEL_KEYWORDS = [
        'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash', 
        'gemini-flash', 'gemini-pro', 'gemini-exp', 'gemma-3'
    ]
    EXCLUDED_MODEL_KEYWORDS = [
        'embedding', 'tts', 'image', 'audio', 'live', 'computer', 'robotics'
    ]

# Configure page
st.set_page_config(
    page_title="AI Guessing Game",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)


class AIGuessingGame:
    def __init__(self, api_key=None, model_name='gemini-1.5-flash'):
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        if api_key:
            self.setup_gemini(api_key, model_name)

    def setup_gemini(self, api_key, model_name='gemini-1.5-flash'):
        """Initialize Gemini API"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
            return True
        except Exception as e:
            st.error(f"❌ Error setting up Gemini API: {str(e)}")
            return False

    def generate_mystery_statement(self, difficulty: str = GameConfig.DEFAULT_DIFFICULTY) -> Tuple[str, str]:
        """Generate a mystery statement and its answer"""
        
        fallback_statements = self._get_fallback_statements()
        
        # Use fallback if quota exhausted or no API
        if self._should_use_fallback():
            return random.choice(fallback_statements.get(difficulty, fallback_statements[GameConfig.DEFAULT_DIFFICULTY]))

        # Track API calls to prevent quota exhaustion
        if st.session_state.get('api_call_count', 0) > GameConfig.MAX_API_CALLS_PER_SESSION:
            st.session_state.use_fallback = True
            st.warning("🔄 Switching to offline mode to preserve API quota")
            return random.choice(fallback_statements.get(difficulty, fallback_statements[GameConfig.DEFAULT_DIFFICULTY]))

        prompt = f"""
        Generate a mystery riddle for {difficulty} difficulty.
        
        Format:
        {{
            "statement": "riddle text",
            "answer": "answer"
        }}
        
        Example: {{"statement": "I have keys but no locks", "answer": "Keyboard"}}
        """

        try:
            response = self.model.generate_content(prompt)
            st.session_state.api_call_count = st.session_state.get('api_call_count', 0) + 1
            response_text = response.text.strip()
            
            # Try to extract JSON if it's embedded in other text
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_text = response_text[start:end]
            else:
                json_text = response_text
            
            result = json.loads(json_text)
            
            # Validate that we have the required keys
            if "statement" not in result or "answer" not in result:
                raise ValueError("Missing required keys in JSON response")
                
            return result["statement"], result["answer"]
            
        except (json.JSONDecodeError, Exception):
            st.warning("🔄 API response error, using offline riddle")
            return random.choice(fallback_statements.get(difficulty, fallback_statements[GameConfig.DEFAULT_DIFFICULTY]))

    def _get_fallback_statements(self) -> Dict[str, List[Tuple[str, str]]]:
        """Get fallback statements organized by difficulty"""
        return {
            "easy": [
                ("I am yellow and curved, monkeys love to eat me", "Banana"),
                ("I have four legs and bark, I am man's best friend", "Dog"),
                ("I am red and round, I keep the doctor away", "Apple"),
                ("I am white and cold, I fall from the sky in winter", "Snow"),
                ("I have wings and can fly, I tweet in the morning", "Bird"),
            ],
            "medium": [
                ("I am made of paper but hold no words, I protect what matters most to you", "Envelope"),
                ("I have a face but no eyes, hands but cannot clap, I tell you something important every second", "Clock"),
                ("I am full of holes but still hold water", "Sponge"),
                ("I can be cracked, I can be made, I can be told, I can be played", "Joke"),
                ("I have cities, but no houses. I have mountains, but no trees. I have water, but no fish", "Map"),
                ("I have keys but no locks, I have space but no room", "Keyboard"),
                ("I can fly without wings, I can cry without eyes", "Cloud"),
            ],
            "hard": [
                ("I am not a season, yet I fall. I am not a musician, yet I have scales", "Waterfall"),
                ("I speak without a mouth and hear without ears, I have no body but come alive with wind", "Echo"),
                ("I am always hungry and must be fed, the finger I touch will soon turn red", "Fire"),
                ("I have a crown but am not a king, I have roots but am not a tree", "Tooth"),
                ("I am taken from a mine and shut in a wooden case, from which I am never released", "Pencil Lead"),
            ]
        }

    def _simple_score(self, guess: str, answer: str) -> int:
        """Simple scoring algorithm for fallback mode"""
        guess_lower = guess.lower().strip()
        answer_lower = answer.lower().strip()
        
        if guess_lower == answer_lower:
            return GameConfig.SCORE_PERFECT
        elif guess_lower in answer_lower or answer_lower in guess_lower:
            return GameConfig.SCORE_CLOSE
        elif any(word in answer_lower.split() for word in guess_lower.split() if len(word) > 2):
            return GameConfig.SCORE_RELATED
        else:
            return GameConfig.SCORE_DISTANT

    def _should_use_fallback(self) -> bool:
        """Check if we should use fallback scoring"""
        return (not self.model or 
                st.session_state.get('use_fallback', False) or 
                st.session_state.get('api_call_count', 0) > GameConfig.MAX_API_CALLS_PER_SESSION)

    def score_multiple_guesses(self, guesses: List[str], answer: str, statement: str) -> List[int]:
        """Score multiple players' guesses in a single API call to save quota"""
        
        # Use simple scoring if in fallback mode or quota exhausted
        if self._should_use_fallback():
            return [self._simple_score(guess, answer) for guess in guesses]

        # Create prompt for multiple guesses
        guess_lines = '\n'.join([f"Guess{i+1}: \"{guess}\"" for i, guess in enumerate(guesses)])
        
        prompt = f"""
        Score all guesses (0-100):
        Answer: "{answer}"
        {guess_lines}
        
        Rules: 100=exact, 80=close, 60=related, 40=somewhat, 20=distant, 0=unrelated
        
        Return: score1,score2,score3... (e.g., "85,65,90")
        """

        try:
            response = self.model.generate_content(prompt)
            st.session_state.api_call_count = st.session_state.get('api_call_count', 0) + 1
            scores_text = response.text.strip()
            scores = [int(s.strip()) for s in scores_text.split(',')]
            if len(scores) == len(guesses):
                return [max(GameConfig.SCORE_MIN, min(GameConfig.SCORE_MAX, score)) for score in scores]
            else:
                raise ValueError("Invalid score format")
        except Exception:
            return [self._simple_score(guess, answer) for guess in guesses]

    def score_guess(self, guess: str, answer: str, statement: str = "") -> int:
        """Score a single player's guess (legacy method for compatibility)"""
        scores = self.score_multiple_guesses([guess], answer, statement)
        return scores[0]


    def score_guesses(self, guess1: str, guess2: str, answer: str, statement: str) -> Tuple[int, int]:
        """Score both players' guesses (legacy method for backward compatibility)"""
        scores = self.score_multiple_guesses([guess1, guess2], answer, statement)
        return scores[0], scores[1]


def get_available_models():
    """Get list of available Gemini models from API (cached)"""
    # Cache the models in session state to avoid repeated API calls
    if 'cached_models' not in st.session_state:
        try:
            # Try to get models from API if key is available
            api_key = st.session_state.get('gemini_api_key', '')
            if api_key and api_key != 'your_gemini_api_key_here':
                genai.configure(api_key=api_key)
                models = genai.list_models()
                
                # Filter models that support generateContent
                suitable_models = {}
                for model in models:
                    if 'generateContent' in model.supported_generation_methods:
                        model_id = model.name.replace('models/', '')
                        # Filter using config constants
                        if (any(keyword in model_id.lower() for keyword in GameConfig.SUPPORTED_MODEL_KEYWORDS) and 
                            not any(exclude in model_id.lower() for exclude in GameConfig.EXCLUDED_MODEL_KEYWORDS)):
                            # Create user-friendly display name
                            display_name = f"{model.display_name}"
                            if model.input_token_limit >= 1000000:
                                display_name += " (1M+ tokens)"
                            elif model.input_token_limit >= 100000:
                                display_name += f" ({model.input_token_limit//1000}K tokens)"
                            
                            suitable_models[model_id] = display_name
                
                st.session_state.cached_models = suitable_models
            else:
                # Fallback to default models if no API key
                st.session_state.cached_models = get_fallback_models()
        except Exception:
            # If API call fails, use fallback models
            st.session_state.cached_models = get_fallback_models()
    
    return st.session_state.cached_models

def get_fallback_models():
    """Fallback models when API is not available"""
    return {
        "gemini-2.5-flash": "Gemini 2.5 Flash (Recommended - Fast & Cost-Effective)",
        "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite (Ultra Fast, Lowest Cost)",
        "gemini-2.5-pro": "Gemini 2.5 Pro (High Quality)",
        "gemini-2.0-flash": "Gemini 2.0 Flash (Balanced Performance)",
        "gemini-flash-latest": "Gemini Flash Latest (Auto-Updated)",
        "gemini-pro-latest": "Gemini Pro Latest (Auto-Updated)",
        "gemini-exp-1206": "Gemini Experimental (Advanced Features)"
    }

def get_model_cost_info():
    """Get cost and performance info for models"""
    return {
        # Gemini 2.5 models (latest and most capable)
        "gemini-2.5-flash": {"cost": "Low", "speed": "Fast", "quality": "High", "recommended": True},
        "gemini-2.5-flash-lite": {"cost": "Lowest", "speed": "Fastest", "quality": "Good", "recommended": True},
        "gemini-2.5-pro": {"cost": "Medium", "speed": "Medium", "quality": "Highest", "recommended": False},
        
        # Gemini 2.0 models
        "gemini-2.0-flash": {"cost": "Low", "speed": "Fast", "quality": "Good", "recommended": False},
        "gemini-2.0-flash-lite": {"cost": "Lowest", "speed": "Fastest", "quality": "Good", "recommended": False},
        "gemini-2.0-pro-exp": {"cost": "Medium", "speed": "Medium", "quality": "High", "recommended": False},
        
        # Latest aliases (auto-updated)
        "gemini-flash-latest": {"cost": "Low", "speed": "Fast", "quality": "High", "recommended": True},
        "gemini-pro-latest": {"cost": "Medium", "speed": "Medium", "quality": "Highest", "recommended": False},
        "gemini-flash-lite-latest": {"cost": "Lowest", "speed": "Fastest", "quality": "Good", "recommended": True},
        
        # Experimental models
        "gemini-exp-1206": {"cost": "High", "speed": "Medium", "quality": "Highest", "recommended": False},
        "learnlm-2.0-flash-experimental": {"cost": "Medium", "speed": "Fast", "quality": "High", "recommended": False},
        
        # Gemma models (open source)
        "gemma-3-1b-it": {"cost": "Lowest", "speed": "Fastest", "quality": "Basic", "recommended": False},
        "gemma-3-4b-it": {"cost": "Lowest", "speed": "Fastest", "quality": "Good", "recommended": False},
        "gemma-3-12b-it": {"cost": "Low", "speed": "Fast", "quality": "Good", "recommended": False},
        "gemma-3-27b-it": {"cost": "Low", "speed": "Medium", "quality": "High", "recommended": False},
    }
def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'game_started': False,
        'current_round': 1,
        'player_scores': {},
        'player_names': ['Player 1', 'Player 2'],
        'num_players': 2,
        'current_statement': "",
        'current_answer': "",
        'round_complete': False,
        'game_complete': False,
        'gemini_api_key': "",
        'api_key_valid': False,
        'use_fallback': False,
        'api_call_count': 0,
        'selected_model': GameConfig.DEFAULT_MODEL
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# UI Component Functions
def render_api_configuration():
    """Render API key configuration section"""
    st.subheader("🔑 Gemini API Configuration")
    api_key = st.text_input(
        "Enter your Gemini API Key:",
        type="password",
        value=st.session_state.gemini_api_key,
        help="Get your API key from: https://makersuite.google.com/app/apikey"
    )
    
    # Update session state and validate API key
    if api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key
        st.session_state.api_key_valid = False
        # Reset game if API key changes
        if st.session_state.game_started:
            st.session_state.game_started = False
            st.warning("⚠️ Game reset due to API key change")
    
    return api_key


def render_model_selection():
    """Render model selection section"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        available_models = get_available_models()
        model_costs = get_model_cost_info()
        
        selected_model = st.selectbox(
            "🤖 Select AI Model:",
            options=list(available_models.keys()),
            format_func=lambda x: available_models[x],
            index=list(available_models.keys()).index(st.session_state.selected_model) 
                  if st.session_state.selected_model in available_models 
                  else 0,
            help="Choose the AI model based on your needs and budget",
            disabled=st.session_state.game_started
        )
    
    with col2:
        if st.button("🔄", help="Refresh model list", disabled=st.session_state.game_started):
            if 'cached_models' in st.session_state:
                del st.session_state.cached_models
            st.rerun()
        
        # Show model count
        model_count = len(available_models)
        st.caption(f"{model_count} models")
    
    # Show model info
    if selected_model in model_costs:
        cost_info = model_costs[selected_model]
        cols = st.columns(4)
        with cols[0]:
            st.metric("💰 Cost", cost_info["cost"])
        with cols[1]:
            st.metric("⚡ Speed", cost_info["speed"])
        with cols[2]:
            st.metric("🎯 Quality", cost_info["quality"])
        with cols[3]:
            if cost_info["recommended"]:
                st.success("✅ Recommended")
            else:
                st.info("💡 Advanced")
    else:
        st.info("ℹ️ Model performance data not available")
    
    # Update session state if model changed
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.api_key_valid = False
        # Reset game if model changes during play
        if st.session_state.game_started:
            st.session_state.game_started = False
            st.warning("⚠️ Game reset due to model change")
    
    return selected_model


def render_api_status():
    """Render API status and testing"""
    api_key = st.session_state.gemini_api_key
    selected_model = st.session_state.selected_model
    available_models = get_available_models()
    
    # Test API key if provided
    if api_key and not st.session_state.api_key_valid:
        if st.button("🔍 Test API Key"):
            try:
                genai.configure(api_key=api_key)
                test_model = genai.GenerativeModel(selected_model)
                # Use a very short prompt to minimize token usage
                test_response = test_model.generate_content("Hi")
                if test_response.text:
                    st.session_state.api_key_valid = True
                    st.success(f"✅ API key is valid with {available_models[selected_model]}!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error with {available_models[selected_model]}: {str(e)}")
                st.session_state.api_key_valid = False
    
    if st.session_state.api_key_valid:
        st.success(f"✅ {available_models[selected_model]}")
        # Show API usage stats
        api_calls = st.session_state.get('api_call_count', 0)
        st.info(f"🔢 API calls used: {api_calls}/{GameConfig.MAX_API_CALLS_PER_SESSION}")
        if api_calls > 15:
            st.warning("⚠️ Approaching API limit")
        if st.session_state.get('use_fallback', False):
            st.info("🔄 Using offline mode")
    elif api_key:
        st.info("🔍 Click 'Test API Key' to validate")
    else:
        st.warning("⚠️ Please enter your Gemini API key")


def render_game_settings():
    """Render game configuration settings"""
    if not st.session_state.api_key_valid:
        st.info("🔑 Please configure a valid API key to access game settings")
        return None, None, None

    total_rounds = st.selectbox("Select number of rounds:", GameConfig.DEFAULT_ROUND_OPTIONS,
                                index=0 if not st.session_state.game_started else None,
                                disabled=st.session_state.game_started)

    difficulty = st.selectbox("Difficulty:", GameConfig.DIFFICULTY_OPTIONS,
                              index=GameConfig.DIFFICULTY_OPTIONS.index(GameConfig.DEFAULT_DIFFICULTY) 
                              if not st.session_state.game_started else None,
                              disabled=st.session_state.game_started)

    # Player management
    st.subheader("👥 Players")
    
    if not st.session_state.game_started:
        # Number of players selector
        num_players = st.slider(
            "Number of players:", 
            min_value=GameConfig.MIN_PLAYERS, 
            max_value=GameConfig.MAX_PLAYERS, 
            value=st.session_state.num_players,
            disabled=st.session_state.game_started
        )
        
        # Update number of players if changed
        if num_players != st.session_state.num_players:
            st.session_state.num_players = num_players
            # Adjust player names list
            current_names = st.session_state.player_names
            if len(current_names) < num_players:
                # Add new players
                for i in range(len(current_names), num_players):
                    current_names.append(f"Player {i + 1}")
            elif len(current_names) > num_players:
                # Remove excess players
                current_names = current_names[:num_players]
            st.session_state.player_names = current_names
        
        # Player name inputs
        player_names = []
        for i in range(num_players):
            default_name = st.session_state.player_names[i] if i < len(st.session_state.player_names) else f"Player {i + 1}"
            name = st.text_input(
                f"Player {i + 1} Name:", 
                value=default_name,
                key=f"player_name_{i}",
                disabled=st.session_state.game_started
            )
            player_names.append(name)
        
        st.session_state.player_names = player_names
    else:
        # Show current players when game is started
        st.write("Current players:")
        for i, name in enumerate(st.session_state.player_names):
            st.write(f"👤 {i + 1}. {name}")
    
    return total_rounds, difficulty, st.session_state.player_names


def render_game_instructions():
    """Render game instructions"""
    st.markdown("## 📋 How to Play")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 Game Rules
        - Play with 2-5 players simultaneously
        - The AI will give you mysterious statements
        - All players guess at the same time
        - AI scores based on how close you are
        - Player with highest total score wins!
        """)

    with col2:
        st.markdown("""
        ### 🏆 Scoring System
        - **100 points**: Perfect answer
        - **80-99 points**: Very close
        - **60-79 points**: Same category
        - **0-59 points**: Not quite there
        """)


def render_player_scores(player_names):
    """Render current player scores"""
    num_players = len(player_names)
    
    if num_players <= 3:
        # For 2-3 players, use columns
        cols = st.columns(num_players)
        for i, name in enumerate(player_names):
            with cols[i]:
                st.subheader(f"👤 {name}")
                score = st.session_state.player_scores.get(name, 0)
                st.metric("Score", f"{score} pts")
    else:
        # For 4-5 players, use a more compact layout
        st.subheader("👥 Current Scores")
        cols = st.columns(2)
        for i, name in enumerate(player_names):
            with cols[i % 2]:
                score = st.session_state.player_scores.get(name, 0)
                st.write(f"👤 **{name}**: {score} pts")


def render_model_comparison():
    """Render model comparison table when not configured"""
    st.markdown("### 🤖 Available AI Models")
    st.markdown("""
    | Model | Cost | Speed | Quality | Best For |
    |-------|------|-------|---------|----------|
    | **Gemini 2.5 Flash** ⭐ | Low | Fast | High | Latest features, best balance |
    | **Gemini 2.5 Flash-Lite** ⭐ | Lowest | Fastest | Good | Quick games, minimal cost |
    | **Gemini Flash Latest** ⭐ | Low | Fast | High | Auto-updated, always current |
    | Gemini 2.5 Pro | Medium | Medium | Highest | Premium quality riddles |
    | Gemini 2.0 Flash | Low | Fast | Good | Proven performance |
    | Gemini Pro Latest | Medium | Medium | Highest | Auto-updated premium |
    | Gemini Experimental | High | Medium | Cutting-edge | Latest research features |
    
    ⭐ = Recommended for this game
    """)
    
    st.info("💡 **Tip**: Start with **Gemini 2.5 Flash** for the best balance of cost, speed, and quality!")


def render_api_instructions():
    """Render API key setup instructions"""
    st.markdown("## 🔑 Getting Your Gemini API Key")
    st.markdown("""
    1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. Sign in with your Google account
    3. Click "Create API Key"
    4. Copy the generated API key
    5. Paste it in the sidebar and click "Test API Key"
    """)


def main():
    initialize_session_state()

    # Header
    st.title("🎯 AI Guessing Game")

    # Sidebar for game setup
    with st.sidebar:
        st.header("🎮 Game Setup")
        
        # API Configuration
        api_key = render_api_configuration()
        
        # Model selection
        selected_model = render_model_selection()
        
        # API status and testing
        render_api_status()
        
        st.divider()
        
        # Initialize game with API key
        if st.session_state.api_key_valid:
            game = AIGuessingGame(api_key, selected_model)
        else:
            game = None

        # Game settings
        total_rounds, difficulty, player_names = render_game_settings()
        
        if st.session_state.api_key_valid and player_names:
            if not st.session_state.game_started:
                if st.button("🚀 Start Game", type="primary"):
                    st.session_state.game_started = True
                    st.session_state.total_rounds = total_rounds
                    st.session_state.difficulty = difficulty
                    st.session_state.player_names = player_names
                    st.session_state.player_scores = {name: 0 for name in player_names}
                    # Reset API call count for new game
                    st.session_state.api_call_count = 0
                    st.session_state.use_fallback = False
                    st.rerun()
            else:
                if st.button("🔄 New Game"):
                    # Reset all session state except API key
                    api_key_backup = st.session_state.gemini_api_key
                    api_valid_backup = st.session_state.api_key_valid
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.session_state.gemini_api_key = api_key_backup
                    st.session_state.api_key_valid = api_valid_backup
                    st.session_state.api_call_count = 0  # Reset for new game
                    st.session_state.use_fallback = False
                    st.rerun()
        else:
            render_model_comparison()

    # Main game area - show different content based on API key status
    if not st.session_state.api_key_valid:
        st.info("👈 Please enter and validate your Gemini API key in the sidebar to start playing!")
        render_api_instructions()
        render_game_instructions()
        return  # Exit early if no valid API key

    # Rest of the game logic (only runs if API key is valid)
    if not st.session_state.game_started:
        st.info("👈 Configure your game settings in the sidebar and click 'Start Game' to begin!")
        render_game_instructions()
    else:
        # Game header with round info
        st.header(f"🎪 Round {st.session_state.current_round} of {st.session_state.total_rounds}")

        # Display current scores
        player_names = st.session_state.player_names
        render_player_scores(player_names)

        # Generate new statement for the round
        if not st.session_state.current_statement and not st.session_state.game_complete:
            with st.spinner("🤖 AI is thinking of something mysterious..."):
                statement, answer = game.generate_mystery_statement(st.session_state.difficulty)
                st.session_state.current_statement = statement
                st.session_state.current_answer = answer
                st.session_state.round_complete = False

        # Display AI statement
        if st.session_state.current_statement and not st.session_state.game_complete:
            st.subheader("🤖 AI Host Says:")
            st.info(f'"{st.session_state.current_statement}"')

            # Player input form (single form for all players)
            if not st.session_state.round_complete:
                with st.form(f"guess_form_{st.session_state.current_round}"):
                    st.subheader("🎯 Enter Your Guesses")
                    
                    # Create columns based on number of players
                    num_players = len(player_names)
                    if num_players <= 3:
                        cols = st.columns(num_players)
                    else:
                        # For 4-5 players, use 2 rows
                        cols_row1 = st.columns(min(3, num_players))
                        if num_players > 3:
                            cols_row2 = st.columns(num_players - 3)
                            cols = list(cols_row1) + list(cols_row2)
                        else:
                            cols = cols_row1
                    
                    player_guesses = {}
                    
                    for i, name in enumerate(player_names):
                        if i < len(cols):
                            with cols[i]:
                                st.write(f"**{name}**")
                                guess = st.text_input(
                                    "Your guess:",
                                    key=f"p{i}_guess_{st.session_state.current_round}",
                                    type="password",
                                    label_visibility="collapsed"
                                )
                                player_guesses[name] = guess

                    # Submit button
                    submit = st.form_submit_button("Submit All Guesses", type="primary")
                    
                    # Show status of guesses
                    filled_guesses = sum(1 for guess in player_guesses.values() if guess)
                    total_players = len(player_names)
                    
                    if filled_guesses == total_players:
                        st.success(f"✅ All {total_players} players have entered their guesses! Ready to submit.")
                    elif filled_guesses > 0:
                        remaining = [name for name, guess in player_guesses.items() if not guess]
                        st.warning(f"⏳ Waiting for: {', '.join(remaining)}")
                    else:
                        st.info(f"📝 All {total_players} players need to enter their guesses before submitting.")

                if submit:
                    empty_guesses = [name for name, guess in player_guesses.items() if not guess]
                    if empty_guesses:
                        st.error(f"❌ The following players must enter their guesses: {', '.join(empty_guesses)}")
                    else:
                        with st.spinner("🤖 AI is scoring all guesses..."):
                            # Score all guesses in a single API call
                            guesses_list = [player_guesses[name] for name in player_names]
                            scores = game.score_multiple_guesses(
                                guesses_list,
                                st.session_state.current_answer,
                                st.session_state.current_statement
                            )

                            # Update scores
                            for i, name in enumerate(player_names):
                                st.session_state.player_scores[name] += scores[i]

                            # Store round results
                            st.session_state.last_round_results = {
                                'guesses': player_guesses,
                                'scores': {name: scores[i] for i, name in enumerate(player_names)},
                                'correct_answer': st.session_state.current_answer
                            }

                            st.session_state.round_complete = True
                            st.rerun()

            # Show round results
            if st.session_state.round_complete:
                results = st.session_state.last_round_results

                st.markdown("### 📊 Round Results")

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.info(f"🎯 **Correct Answer:** {results['correct_answer']}")

                # Display results for all players
                num_players = len(player_names)
                if num_players <= 2:
                    cols = st.columns(num_players)
                    for i, name in enumerate(player_names):
                        with cols[i]:
                            st.success(f"""
                            **{name}'s Result:**
                            - Guess: "{results['guesses'][name]}"
                            - Score: {results['scores'][name]} points
                            """)
                else:
                    # For more than 2 players, use a more compact layout
                    st.markdown("#### 🏆 Player Results")
                    
                    # Sort players by score for this round (descending)
                    sorted_players = sorted(player_names, key=lambda x: results['scores'][x], reverse=True)
                    
                    for i, name in enumerate(sorted_players):
                        score = results['scores'][name]
                        guess = results['guesses'][name]
                        
                        # Add medal emojis for top performers
                        if i == 0 and score > 0:
                            emoji = "🥇"
                        elif i == 1 and score > 0:
                            emoji = "🥈"
                        elif i == 2 and score > 0:
                            emoji = "🥉"
                        else:
                            emoji = "👤"
                        
                        st.write(f"{emoji} **{name}**: \"{guess}\" → **{score} points**")

                # Next round or end game
                if st.session_state.current_round < st.session_state.total_rounds:
                    if st.button("➡️ Next Round", type="primary"):
                        st.session_state.current_round += 1
                        st.session_state.current_statement = ""
                        st.session_state.current_answer = ""
                        st.session_state.round_complete = False
                        st.rerun()
                else:
                    st.session_state.game_complete = True
                    st.rerun()

        # Game completion
        if st.session_state.game_complete:
            # Determine winner(s)
            final_scores = st.session_state.player_scores
            max_score = max(final_scores.values())
            winners = [name for name, score in final_scores.items() if score == max_score]

            # Winner announcement
            st.markdown("### 🎉 Game Complete!")
            
            if len(winners) == 1:
                st.success(f"🏆 Congratulations {winners[0]}!")
                st.write(f"Champion with {max_score} points!")
                st.write("You are the Guessing Master! 👑")
            else:
                st.success(f"🤝 Amazing! It's a {len(winners)}-way Tie!")
                st.write(f"Winners: {', '.join(winners)}")
                st.write(f"All tied with {max_score} points!")
                st.write("You're all champions! 🏆")

            # Final score summary
            st.subheader("📈 Final Leaderboard")
            
            # Sort players by final score (descending)
            sorted_final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Display leaderboard
            for i, (name, score) in enumerate(sorted_final):
                if i == 0:
                    emoji = "🥇"
                elif i == 1:
                    emoji = "🥈" 
                elif i == 2:
                    emoji = "🥉"
                else:
                    emoji = "👤"
                
                st.metric(label=f"{emoji} {name}", value=f"{score} points")

            st.balloons()


if __name__ == "__main__":
    main()
