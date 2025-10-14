# 🎯 AI-Powered Guessing Game

A fun, interactive guessing game built with Streamlit where two players compete to guess what the AI is describing!

## 🌟 Features

- **Dual Player Mode**: Two players play simultaneously
- **AI Host**: Powered by Google's Gemini 2.0 Flash model
- **Smart Scoring**: AI evaluates how close your guesses are
- **Flexible Rounds**: Choose 3, 5, or 7 rounds
- **Clean UI**: Beautiful, responsive Streamlit interface
- **Difficulty Levels**: Easy, Medium, and Hard modes

## 🚀 Setup Instructions

### 1. Clone/Download the Project
```bash
cd /path/to/GuessStreamLit
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get Your Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy your API key

### 4. Configure Environment Variables
1. Open the `.env` file
2. Replace `your_gemini_api_key_here` with your actual API key:
   ```
   GEMINI_API_KEY=AIzaSyC1234567890abcdefghijk
   ```

### 5. Run the Application
```bash
streamlit run app.py
```

The game will open in your web browser at `http://localhost:8501`

## 🎮 How to Play

1. **Setup Game**: 
   - Choose number of rounds (3, 5, or 7)
   - Select difficulty level
   - Enter player names
   - Click "Start Game"

2. **Playing**:
   - AI will present a mysterious statement
   - Both players enter their guesses simultaneously 
   - AI scores each guess based on accuracy (0-100 points)
   - Continue for all rounds

3. **Winning**:
   - Player with the highest total score wins!
   - Ties are celebrated equally 🎉

## 🏆 Scoring System

- **100 points**: Perfect answer or exact match
- **80-99 points**: Very close, minor differences
- **60-79 points**: Related concept, same category  
- **40-59 points**: Somewhat related
- **20-39 points**: Distantly related
- **0-19 points**: Not related at all

## 🛠️ Technical Details

- **Frontend**: Streamlit with custom CSS
- **AI Model**: Google Gemini 2.0 Flash 
- **Languages**: Python 3.7+
- **Dependencies**: 
  - `streamlit==1.28.1`
  - `google-generativeai==0.3.2`
  - `python-dotenv==1.0.0`

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile
- **Clean Interface**: Minimalist, intuitive design
- **Visual Feedback**: Score displays, progress indicators
- **Animations**: Celebration effects for winners

## 🔧 Troubleshooting

### API Key Issues
- Make sure your `.env` file has the correct API key
- Verify the API key is active in Google AI Studio
- Check for any billing issues in your Google Cloud account

### Installation Issues
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Streamlit Issues
```bash
streamlit cache clear
```

## 📝 Example Game Flow

1. **AI Statement**: "I am round and bright, I rise and set, bringing warmth to your day"
2. **Player 1 Guess**: "Sun" → Score: 100 points
3. **Player 2 Guess**: "Moon" → Score: 60 points (related but not exact)

## 🔮 Future Enhancements

- [ ] Multiplayer support (3+ players)
- [ ] Different game modes (categories, time limits)
- [ ] Leaderboard system
- [ ] Sound effects and music
- [ ] Mobile app version

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the game!

---

**Have fun guessing! 🎯✨**
