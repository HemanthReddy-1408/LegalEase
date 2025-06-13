from langchain.memory import ConversationBufferWindowMemory
import json
import os
from datetime import datetime

def get_memory(session_id=None, max_token_limit=2000):
    """
    Creates conversation memory with better configuration.
    Uses window memory to prevent token overflow.
    
    Args:
        session_id: Optional session ID for persistence
        max_token_limit: Maximum tokens to keep in memory
    
    Returns:
        ConversationBufferWindowMemory: Configured memory object
    """
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,  # Keep last 5 exchanges
        max_token_limit=max_token_limit
    )
    
    # Load previous session if available
    if session_id:
        memory_file = f"sessions/{session_id}_memory.json"
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                    # Restore memory from saved data
                    for msg_data in memory_data:
                        if msg_data.get('type') == 'human':
                            memory.chat_memory.add_user_message(msg_data['content'])
                        elif msg_data.get('type') == 'ai':
                            memory.chat_memory.add_ai_message(msg_data['content'])
                print(f"📋 Loaded session memory: {session_id}")
            except Exception as e:
                print(f"⚠️ Failed to load memory for session {session_id}: {e}")
    
    return memory

def save_memory(memory, session_id):
    """
    Save memory to disk for persistence
    
    Args:
        memory: Memory object to save
        session_id: Session identifier
    """
    try:
        os.makedirs("sessions", exist_ok=True)
        memory_file = f"sessions/{session_id}_memory.json"
        
        # Convert memory to serializable format
        memory_data = []
        for message in memory.chat_memory.messages:
            if hasattr(message, 'type'):
                memory_data.append({
                    'type': message.type,
                    'content': message.content,
                    'timestamp': datetime.now().isoformat()
                })
        
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Memory saved for session: {session_id}")
    except Exception as e:
        print(f"❌ Failed to save memory for session {session_id}: {e}")

def clear_session_memory(session_id):
    """
    Clear memory for a specific session
    
    Args:
        session_id: Session identifier to clear
    """
    try:
        memory_file = f"sessions/{session_id}_memory.json"
        if os.path.exists(memory_file):
            os.remove(memory_file)
            print(f"🧹 Cleared memory for session: {session_id}")
    except Exception as e:
        print(f"❌ Failed to clear memory for session {session_id}: {e}")

def get_session_list():
    """
    Get list of available sessions
    
    Returns:
        list: List of session IDs
    """
    try:
        if not os.path.exists("sessions"):
            return []
        
        sessions = []
        for filename in os.listdir("sessions"):
            if filename.endswith("_memory.json"):
                session_id = filename.replace("_memory.json", "")
                sessions.append(session_id)
        
        return sessions
    except Exception as e:
        print(f"❌ Failed to get session list: {e}")
        return []