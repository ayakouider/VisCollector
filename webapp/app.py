from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import sys
import json
import threading
import uuid
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

try:
    from agent.agent import run_agent
    from agent.data_manager import get_stats,print_stats
    import google.generativeai as genai
    Gemini_available = True
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app,cors_allowed_origins="*")

jobs = {}
active_jobs = {}

gemini_api_key = os.getenv("GEMINI_API_KEY")
system_prompt = """You are an AI assistant for a Vision Data Collection Agent that processes YouTube videos.

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

5. CHAT - Just conversation (no action needed)
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
- Be conversational in chat responses but concise"""

chatbot_session = None

def init_chatbot():
    global chatbot_session
    if not gemini_api_key or not Gemini_available:
        return False
    
    try:
        genai.configure(api_key=gemini_api_key)
        try:
            available_models = genai.list_models()
            model_names = [m.name for m in available_models if 'generateContent' in m.supported_generation_methods]
            print(f"Available models: {[m.split('/')[-1] for m in model_names[:5]]}")
        except:
            pass
        
        gen_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048
        }

        models_to_try = [
            "gemini-2.5-flash",          
            "gemini-2.0-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash-001",
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash"
        ]
        
        for model_name in models_to_try:
            try:
                print(f"Trying model: {model_name}")
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=gen_config,
                )
                
                # Start chat and send system prompt as first message
                chatbot_session = model.start_chat(history=[])
                chatbot_session.send_message(system_prompt)
                
                print(f"✓ Successfully initialized with model: {model_name}")
                return True
            except Exception as e:
                print(f"  Failed with {model_name}: {str(e)[:100]}")
                continue
        
        print("✗ All model initialization attempts failed")
        return False
    except Exception as e:
        print(f"Failed to init chatbot: {e}")
        return False
        
   
def create_job(job_type:str,params:dict):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "type": job_type,
        "params": params,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "progress": 0,
        "current_step": None,
        "result": None,
        "error": None
    }
    return job_id

def update_job(job_id:str,updates:dict):
    if job_id in jobs:
        jobs[job_id].update(updates)
        socketio.emit('job_update', jobs[job_id], room=job_id)

def run_agent_job(job_id:str,params:dict):
    try:
        update_job(job_id, {"status": "running", "progress": 5, "current_step": "Initializing"})
        socketio.emit("log",{
        'job_id':job_id,
        'message': f"Starting job: {params.get('query') or params.get('urls')}"
        })

        update_job(job_id, {"progress": 10, "current_step": "Discovering videos"})
        socketio.emit('log', {'job_id': job_id, 'message': '🔍 Searching for videos...'})

        results = run_agent(
            query = params.get('query'),
            urls = params.get('urls'),
            channel_url= params.get('channel_url'),
            max_results= params.get("max_results",3),
            skip_blur= params.get('skip_blur',False),
            skip_dedup= params.get('skip_dedup',False),
            dry_run= params.get('dry_run',False),
            smart_search=True,
            min_duration=30,      # 15 seconds minimum
            max_duration=1200,     # 3 minutes maximum (short videos only)
            sort_by="relevance"
        )

        update_job(job_id, {"progress": 30, "current_step": "Downloading videos"})
        socketio.emit('log', {'job_id': job_id, 'message': '⬇️  Downloading videos...'})

        update_job(job_id, {"progress": 50, "current_step": "Extracting frames"})
        socketio.emit('log', {'job_id': job_id, 'message': '🎞️  Extracting frames from videos...'})
        
        update_job(job_id, {"progress": 70, "current_step": "Detecting faces"})
        socketio.emit('log', {'job_id': job_id, 'message': '👤 Detecting and extracting faces...'})
        
        update_job(job_id, {"progress": 85, "current_step": "Filtering quality"})
        socketio.emit('log', {'job_id': job_id, 'message': '🔬 Filtering blur and duplicates...'})
        

        update_job(job_id,{
            "status":"completed",
            "progress":100,
            "results": results
        })

        socketio.emit('log',{
            'job_id':job_id,
            'message':f"Job completed successfully! Processed {len(results)} videos."
        })

    except Exception as e:
        update_job(job_id,{
            'status':"failed",
            'error': str(e)
        })

        socketio.emit('log',{
            'job_id':job_id,
            'message': f"Job failed: {str(e)}",
            'level': 'error'
        })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status":"ok",
        "gemini_available":Gemini_available and gemini_api_key is not None,
        "active_jobs": len([j for j in jobs.values() if j['status'] == 'running'])

    })

@app.route('/api/chat',methods=['POST'])
def chat():
    data = request.json
    message = data.get('message','')

    if not chatbot_session:
        if not init_chatbot():
            return jsonify({
                "action": "chat",
                "response": "AI assistant unavailable. Please set GEMINI_API_KEY environment variable."
            })
        
    try:
        response = chatbot_session.send_message(message)
        text = response.text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        command = json.loads(text)
        return jsonify(command)
    
    except json.JSONDecodeError:
        return jsonify({
            "action": "chat",
            "response": "Sorry, I had trouble understanding that. Can you rephrase?"
        })
    except Exception as e:
        return jsonify({
            "action":"chat",
            "response":f"Error: {str(e)}"
        }),500
    
@app.route('/api/process',methods = ['POST'])
def process():
    data = request.json
    job_id = create_job('process',data)

    thread = threading.Thread(
        target=run_agent_job,
        args=(job_id,data)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'job_id':job_id,
        'status':'queued'
    })

@app.route('/api/stats',methods=['GET'])
def stats():
    try:
        stats_data = get_stats()
        return jsonify(stats_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/jobs',methods=['GET'])
def get_jobs():
    return jsonify(list(jobs.values()))

@app.route('/api/jobs/<job_id>',methods=['GET'])
def get_job(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected to Vision Data Agent'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('subscribe_job')
def handle_subscribe(data):
    job_id = data.get('job_id')
    if job_id:
        from flask_socketio import join_room
        join_room(job_id)
        emit('subscribed', {'job_id': job_id})

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  VISION DATA AGENT — WEB API SERVER")
    print("=" * 60)
    print(f"\n  Backend running on: http://localhost:5000")
    print(f"  Gemini AI: {'✓ Enabled' if gemini_api_key else '✗ Disabled (set GEMINI_API_KEY)'}")
    print(f"\n  API Endpoints:")
    print(f"    POST /api/chat       - AI chatbot")
    print(f"    POST /api/process    - Start job")
    print(f"    GET  /api/stats      - Dataset stats")
    print(f"    GET  /api/jobs       - List all jobs")
    print("\n" + "=" * 60 + "\n")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
