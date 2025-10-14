import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import time
from typing import Dict, List, Tuple

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="AI Guessing Game",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class AIGuessingGame:
    def __init__(self):
        self.setup_gemini()
        
    def setup_gemini(self):
        """Initialize Gemini API"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key or api_key == 'your_gemini_api_key_here':
            st.error("🚫 Please set up your Gemini API key in the .env file!")
            st.info("Get your API key from: https://makersuite.google.com/app/apikey")
            st.stop()
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
    
    def generate_mystery_statement(self, difficulty: str = "medium") -> Tuple[str, str]:
        """Generate a mystery statement and its answer"""
        prompt = f"""
        Generate a mysterious statement about a common object, person, place, or concept. 
        The difficulty should be {difficulty}.
        
        Return your response in this exact JSON format:
        {{
            "statement": "A mysterious clue about the subject without revealing what it is",
            "answer": "The actual subject being described"
        }}
        
        Examples:
        - "I am round and bright, I rise and set, bringing warmth to your day" -> "Sun"
        - "I have keys but no locks, I have space but no room" -> "Keyboard"
        - "I can fly without wings, I can cry without eyes" -> "Cloud"
        
        Make it creative but fair to guess.
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text.strip())
            return result["statement"], result["answer"]
        except Exception as e:
            # Fallback statements
            fallbacks = [
                ("I am made of paper but hold no words, I protect what matters most to you", "Envelope"),
                ("I have a face but no eyes, hands but cannot clap, I tell you something important every second", "Clock"),
                ("I am full of holes but still hold water", "Sponge")
            ]
            import random
            return random.choice(fallbacks)
    
    def score_guess(self, guess: str, answer: str, statement: str) -> int:
        """Score a player's guess using AI"""
        prompt = f"""
        Score how close this guess is to the correct answer on a scale of 0-100.
        
        Statement: "{statement}"
        Correct Answer: "{answer}"
        Player's Guess: "{guess}"
        
        Scoring Guidelines:
        - 100 points: Exact match or perfect synonym
        - 80-99 points: Very close, minor spelling/wording differences
        - 60-79 points: Related concept, same category
        - 40-59 points: Somewhat related
        - 20-39 points: Distantly related
        - 0-19 points: Not related at all
        
        Return only the numeric score (0-100).
        """
        
        try:
            response = self.model.generate_content(prompt)
            score = int(response.text.strip())
            return max(0, min(100, score))  # Ensure score is between 0-100
        except Exception:
            # Simple fallback scoring
            guess_lower = guess.lower().strip()
            answer_lower = answer.lower().strip()
            
            if guess_lower == answer_lower:
                return 100
            elif guess_lower in answer_lower or answer_lower in guess_lower:
                return 80
            else:
                return 20

def initialize_session_state():
    """Initialize session state variables"""
    if 'game_started' not in st.session_state:
        st.session_state.game_started = False
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 1
    if 'player_scores' not in st.session_state:
        st.session_state.player_scores = {'Player 1': 0, 'Player 2': 0}
    if 'current_statement' not in st.session_state:
        st.session_state.current_statement = ""
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = ""
    if 'round_complete' not in st.session_state:
        st.session_state.round_complete = False
    if 'game_complete' not in st.session_state:
        st.session_state.game_complete = False

def main():
    initialize_session_state()
    game = AIGuessingGame()
    
    # Header
    st.title("🎯 AI Guessing Game")
    
    # Sidebar for game setup
    with st.sidebar:
        st.header("🎮 Game Setup")
        
        # Game settings
        total_rounds = st.selectbox("Select number of rounds:", [3, 5, 7], 
                                  index=0 if not st.session_state.game_started else None,
                                  disabled=st.session_state.game_started)
        
        difficulty = st.selectbox("Difficulty:", ["easy", "medium", "hard"],
                                index=1 if not st.session_state.game_started else None,
                                disabled=st.session_state.game_started)
        
        # Player names
        player1_name = st.text_input("Player 1 Name:", value="Player 1",
                                   disabled=st.session_state.game_started)
        player2_name = st.text_input("Player 2 Name:", value="Player 2",
                                   disabled=st.session_state.game_started)
        
        if not st.session_state.game_started:
            if st.button("🚀 Start Game", type="primary"):
                st.session_state.game_started = True
                st.session_state.total_rounds = total_rounds
                st.session_state.difficulty = difficulty
                st.session_state.player_names = [player1_name, player2_name]
                st.session_state.player_scores = {player1_name: 0, player2_name: 0}
                st.rerun()
        else:
            if st.button("🔄 New Game"):
                # Reset all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Main game area
    if not st.session_state.game_started:
        st.info("👈 Configure your game settings in the sidebar and click 'Start Game' to begin!")
        
        # Game instructions
        st.markdown("## 📋 How to Play")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Game Rules
            - The AI will give you mysterious statements
            - Both players guess simultaneously
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
        
    else:
        # Game header with round info
        st.header(f"🎪 Round {st.session_state.current_round} of {st.session_state.total_rounds}")
        
        # Display current scores
        col1, col2 = st.columns(2)
        
        with col1:
            player1_name = st.session_state.player_names[0]
            st.subheader(f"👤 {player1_name}")
            st.metric("Score", f"{st.session_state.player_scores[player1_name]} pts")
        
        with col2:
            player2_name = st.session_state.player_names[1]
            st.subheader(f"👤 {player2_name}")
            st.metric("Score", f"{st.session_state.player_scores[player2_name]} pts")
        
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
            
            # Player input forms
            if not st.session_state.round_complete:
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.form(f"player1_form_{st.session_state.current_round}"):
                        st.subheader(f"🎯 {player1_name}'s Guess")
                        player1_guess = st.text_input("Your guess:", key="p1_guess")
                        player1_submit = st.form_submit_button("Submit Guess", type="primary")
                
                with col2:
                    with st.form(f"player2_form_{st.session_state.current_round}"):
                        st.subheader(f"🎯 {player2_name}'s Guess")
                        player2_guess = st.text_input("Your guess:", key="p2_guess")
                        player2_submit = st.form_submit_button("Submit Guess", type="primary")
                
                # Process guesses when both players submit
                if player1_submit and player2_submit and player1_guess and player2_guess:
                    with st.spinner("🤖 AI is scoring your guesses..."):
                        # Score both guesses
                        score1 = game.score_guess(player1_guess, st.session_state.current_answer, st.session_state.current_statement)
                        score2 = game.score_guess(player2_guess, st.session_state.current_answer, st.session_state.current_statement)
                        
                        # Update scores
                        st.session_state.player_scores[player1_name] += score1
                        st.session_state.player_scores[player2_name] += score2
                        
                        # Store round results
                        st.session_state.last_round_results = {
                            'player1_guess': player1_guess,
                            'player2_guess': player2_guess,
                            'score1': score1,
                            'score2': score2,
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
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"""
                    **{player1_name}'s Result:**
                    - Guess: "{results['player1_guess']}"
                    - Score: {results['score1']} points
                    """)
                
                with col2:
                    st.success(f"""
                    **{player2_name}'s Result:**
                    - Guess: "{results['player2_guess']}"
                    - Score: {results['score2']} points
                    """)
                
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
            # Determine winner
            final_scores = st.session_state.player_scores
            player1_score = final_scores[player1_name]
            player2_score = final_scores[player2_name]
            
            if player1_score > player2_score:
                winner = player1_name
                winner_score = player1_score
            elif player2_score > player1_score:
                winner = player2_name
                winner_score = player2_score
            else:
                winner = "It's a tie!"
                winner_score = player1_score
            
            # Winner announcement
            if winner == "It's a tie!":
                st.success("🤝 Amazing! It's a Tie!")
                st.write(f"Both players scored {winner_score} points!")
                st.write("You're both champions! 🏆")
            else:
                st.success(f"🎉 Congratulations {winner}!")
                st.write(f"Winner with {winner_score} points!")
                st.write("You are the Guessing Champion! 👑")
            
            # Final score summary
            st.subheader("📈 Final Score Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label=f"🏆 {player1_name}", value=f"{player1_score} points")
            
            with col2:
                st.metric(label=f"🏆 {player2_name}", value=f"{player2_score} points")
            
            st.balloons()

if __name__ == "__main__":
    main()
