"""Simple file-based storage for conversations."""
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from .config import DATA_DIR


def ensure_data_dir():
    """Ensure the data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)


def get_conversation_path(conversation_id: str) -> str:
    """Get the file path for a conversation."""
    ensure_data_dir()
    return os.path.join(DATA_DIR, f"{conversation_id}.json")


def create_conversation(conversation_id: str) -> Dict[str, Any]:
    """Create a new conversation."""
    conversation = {
        "id": conversation_id,
        "created_at": datetime.now().isoformat(),
        "title": "New Conversation",
        "messages": []
    }
    
    with open(get_conversation_path(conversation_id), 'w') as f:
        json.dump(conversation, f, indent=2)
    
    return conversation


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Get a conversation by ID."""
    path = get_conversation_path(conversation_id)
    
    if not os.path.exists(path):
        return None
    
    with open(path, 'r') as f:
        return json.load(f)


def list_conversations() -> List[Dict[str, Any]]:
    """List all conversations (metadata only)."""
    ensure_data_dir()
    
    conversations = []
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            conversation_id = filename[:-5]
            conv = get_conversation(conversation_id)
            if conv:
                conversations.append({
                    "id": conv["id"],
                    "created_at": conv["created_at"],
                    "title": conv["title"],
                    "message_count": len(conv["messages"])
                })
    
    conversations.sort(key=lambda x: x["created_at"], reverse=True)
    
    return conversations


def update_conversation_title(conversation_id: str, title: str):
    """Update the title of a conversation."""
    conv = get_conversation(conversation_id)
    if conv:
        conv["title"] = title
        with open(get_conversation_path(conversation_id), 'w') as f:
            json.dump(conv, f, indent=2)


def add_user_message(conversation_id: str, content: str):
    """Add a user message to a conversation."""
    conv = get_conversation(conversation_id)
    if conv:
        conv["messages"].append({
            "role": "user",
            "content": content
        })
        with open(get_conversation_path(conversation_id), 'w') as f:
            json.dump(conv, f, indent=2)


def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage3: Dict[str, Any]
):
    """Add an assistant message with all stages to a conversation."""
    conv = get_conversation(conversation_id)
    if conv:
        conv["messages"].append({
            "role": "assistant",
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3
        })
        with open(get_conversation_path(conversation_id), 'w') as f:
            json.dump(conv, f, indent=2)
