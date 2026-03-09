from __future__ import annotations
import argparse
import json
import os
import sys
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent.agent import run_agent
from agent.data_manager import print_stats, get_stats
import google.generativeai as genai

# Try different model names - use the one that works with your API key
GEMINI_MODEL = "gemini-1.5-flash"  # Try without -latest first

SYSTEM_PROMPT = """You are an AI assistant for a Vision Data Collection Agent that processes YouTube videos.

Your job: Parse user requests into structured JSON commands for the agent.

## Available Commands

1. SEARCH - Search YouTube and process videos
   {
     "action": "search",
     "query": "search terms",
     "max_results": 5,
     "skip_blur": false,
     "skip_dedup": false,
     "dry_run": false
   }

2. DOWNLOAD - Process specific YouTube URLs
   {
     "action": "download",
     "urls": ["https://youtube.com/watch?v=XXX"],
     "skip_blur": false,
     "skip_dedup": false,
     "dry_run": false
   }

3. CHANNEL - Process videos from a channel
   {
     "action": "channel",
     "channel_url": "https://youtube.com/@ChannelName",
     "max_results": 10,
     "skip_blur": false,
     "skip_dedup": false,
     "dry_run": false
   }

4. STATS - Show dataset statistics
   {
     "action": "stats"
   }

5. HELP - Show help
   {
     "action": "help"
   }

6. CHAT - Just conversation (no action needed)
   {
     "action": "chat",
     "response": "your friendly response here"
   }

## Important Rules

- ALWAYS respond with valid JSON only, nothing else
- Extract YouTube URLs exactly as provided
- Infer max_results from user's phrasing (e.g., "5 videos" → 5, "a few" → 3, "some" → 5)
- "skip blur" / "no blur filter" → skip_blur: true
- "skip dedup" / "skip duplicates" → skip_dedup: true  
- "preview" / "dry run" / "just show me" → dry_run: true
- For casual conversation, use action: "chat"
- Be conversational in chat responses but concise

## Examples

User: "Search for 5 videos about fake news"
You: {"action": "search", "query": "fake news", "max_results": 5, "skip_blur": false, "skip_dedup": false, "dry_run": false}

User: "Download this: https://youtube.com/watch?v=abc123 but skip the blur filter"
You: {"action": "download", "urls": ["https://youtube.com/watch?v=abc123"], "skip_blur": true, "skip_dedup": false, "dry_run": false}

User: "How many videos have we processed?"
You: {"action": "stats"}

User: "Hey, how's it going?"
You: {"action": "chat", "response": "I'm doing great! Ready to help you collect some vision data. What would you like to do?"}

User: "Get me like 10 videos from this channel https://youtube.com/@Example but just preview them first"
You: {"action": "channel", "channel_url": "https://youtube.com/@Example", "max_results": 10, "skip_blur": false, "skip_dedup": false, "dry_run": true}
"""

class ChatBot:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        
        # Try to list available models to see what's actually available
        try:
            models = genai.list_models()
            available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            print(f"[DEBUG] Available models: {available[:3]}...")  # Show first 3
        except:
            pass
        
        # Configure generation settings
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Try creating the model with proper config
        try:
            self.model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                generation_config=generation_config,
            )
            # Start chat WITHOUT system instruction in constructor
            # We'll send it as first message instead
            self.chat = self.model.start_chat(history=[])
            
            # Send system prompt as first message
            self.chat.send_message(SYSTEM_PROMPT)
            
        except Exception as e:
            print(f"[ERROR] Failed with {GEMINI_MODEL}, trying fallback...")
            # Fallback to older model
            self.model = genai.GenerativeModel(
                model_name="gemini-pro",
                generation_config=generation_config,
            )
            self.chat = self.model.start_chat(history=[])
            self.chat.send_message(SYSTEM_PROMPT)

    def send(self, message: str):
        try:
            response = self.chat.send_message(message)
            text = response.text.strip()
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            command = json.loads(text)
            return command
            
        except json.JSONDecodeError as e:
            print(f"\n⚠️  Chatbot returned invalid JSON: {text}")
            print(f"   Error: {e}")
            return {"action": "chat", "response": "Sorry, I had trouble understanding that. Can you rephrase?"}
        
        except Exception as e:
            print(f"\n❌ Error communicating with Chatbot: {e}")
            return {"action": "chat", "response": "I ran into an error. Please try again."}


def handle_search(query: str, max_results: int, skip_blur: bool, skip_dedup: bool, dry_run: bool):
    print(f"\n🔍 Searching for: '{query}' (max {max_results} videos)")
    if skip_blur:  print("   ⚙️  Blur filtering: DISABLED")
    if skip_dedup: print("   ⚙️  Dedup: DISABLED")
    if dry_run:    print("   ⚙️  Mode: DRY RUN")
    print()

    run_agent(
        query=query,
        max_results=max_results,
        skip_blur=skip_blur,
        skip_dedup=skip_dedup,
        dry_run=dry_run,
    )


def handle_url(urls: list[str], skip_blur: bool, skip_dedup: bool, dry_run: bool):
    print(f"\n📥 Processing {len(urls)} URL(s)")
    if skip_blur:  print("   ⚙️  Blur filtering: DISABLED")
    if skip_dedup: print("   ⚙️  Dedup: DISABLED")
    if dry_run:    print("   ⚙️  Mode: DRY RUN")
    print()
    
    run_agent(
        urls=urls,
        skip_blur=skip_blur,
        skip_dedup=skip_dedup,
        dry_run=dry_run,
    )


def handle_channel(channel_url: str, max_results: int, skip_blur: bool, skip_dedup: bool, dry_run: bool):
    print(f"\n📺 Processing channel: {channel_url} (max {max_results} videos)")
    if skip_blur:  print("   ⚙️  Blur filtering: DISABLED")
    if skip_dedup: print("   ⚙️  Dedup: DISABLED")
    if dry_run:    print("   ⚙️  Mode: DRY RUN")
    print()
    
    run_agent(
        channel_url=channel_url,
        max_results=max_results,
        skip_blur=skip_blur,
        skip_dedup=skip_dedup,
        dry_run=dry_run,
    )


def handle_stats():
    stats = get_stats()
    print_stats()
    return stats


def help():
    print("\n" + "─" * 60)
    print("  VISION DATA AGENT")
    print("─" * 60)
    print("\nTalk to me naturally! I understand:\n")
    print("  • 'Search for videos about [topic]'")
    print("  • 'Download this video: [URL]'")
    print("  • 'Get 10 videos from this channel: [URL]'")
    print("  • 'Show me dataset stats'")
    print("  • 'Skip blur filtering' (add to any command)")
    print("  • 'Dry run' (preview only)")
    print("  • Casual conversation!")
    print("\nType 'quit' or 'exit' to leave.")
    print("─" * 60 + "\n")


def chatbot(api_key: str):
    print("\n" + "═" * 60)
    print("  🤖 VISION DATA AGENT — GEMINI CHATBOT")
    print("═" * 60)
    print("\nInitializing AI assistant...")

    try:
        bot = ChatBot(api_key=api_key)
        print("✓ Chatbot initialized successfully.\n")

    except Exception as e:
        print(f"\n❌ Failed to initialize Chatbot: {e}")
        print("   Check your API key and internet connection.\n")
        return
        
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "bye", "q"]:
                print("\n👋 Goodbye!")
                break

            print("🤔 Thinking...", end="", flush=True)
            command = bot.send(user_input)
            print("\r              \r", end="")  # Clear "Thinking..."

            action = command.get("action")

            if action == "search":
                handle_search(
                    query=command.get("query", ""),
                    max_results=command.get("max_results", 2),
                    skip_blur=command.get("skip_blur", False),
                    skip_dedup=command.get("skip_dedup", False),
                    dry_run=command.get("dry_run", False),
                )

            elif action == "download":
                handle_url(
                    urls=command.get("urls", []),
                    skip_blur=command.get("skip_blur", False),
                    skip_dedup=command.get("skip_dedup", False),
                    dry_run=command.get("dry_run", False),
                )

            elif action == "channel":
                handle_channel(
                    channel_url=command.get("channel_url", ""),
                    max_results=command.get("max_results", 2),
                    skip_blur=command.get("skip_blur", False),
                    skip_dedup=command.get("skip_dedup", False),
                    dry_run=command.get("dry_run", False),
                )

            elif action == "stats":
                stats = handle_stats()
                follow_up = bot.send(f"The stats are: {json.dumps(stats)}. Give a brief 1-sentence summary.")
                if follow_up.get("action") == "chat":
                    print(f"\n💬 Bot: {follow_up.get('response', '')}\n")

            elif action == "help":
                help()
            
            elif action == "chat":
                response = command.get("response", "I'm not sure how to help with that.")
                print(f"\n💬 Bot: {response}\n")

            else:
                print(f"\n❓ Unknown command: {action}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break

        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Vision Data Agent — Gemini-powered chatbot"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("\n❌ ERROR: No Gemini API key provided.\n")
        print("Get one at: https://aistudio.google.com/app/apikey")
        print("\nThen either:")
        print("  1. Set environment variable:")
        print("     Windows: $env:GEMINI_API_KEY='your_key_here'")
        print("     Linux/Mac: export GEMINI_API_KEY=your_key_here")
        print("\n  2. Pass directly:")
        print("     python chatbot_gemini_fixed.py --api_key YOUR_KEY\n")
        sys.exit(1)
    
    chatbot(api_key)


if __name__ == "__main__":
    main()