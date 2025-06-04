"""
会话管理器 - 存储和管理聊天会话记录
支持按会话存储、查询和调取聊天记录
"""
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ChatMessage:
    """聊天消息"""
    id: str
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """从字典创建"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ChatSession:
    """聊天会话"""
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessage]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'session_id': self.session_id,
            'title': self.title,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'messages': [msg.to_dict() for msg in self.messages],
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """从字典创建"""
        return cls(
            session_id=data['session_id'],
            title=data['title'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            messages=[ChatMessage.from_dict(msg) for msg in data['messages']],
            metadata=data.get('metadata', {})
        )


class SessionManager:
    """会话管理器"""
    
    def __init__(self, storage_dir: str = "data/sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[ChatSession] = None
        self.sessions_index_file = self.storage_dir / "sessions_index.json"
        self._load_sessions_index()
    
    def _load_sessions_index(self):
        """加载会话索引"""
        if self.sessions_index_file.exists():
            try:
                with open(self.sessions_index_file, 'r', encoding='utf-8') as f:
                    self.sessions_index = json.load(f)
            except Exception as e:
                print(f"加载会话索引失败: {e}")
                self.sessions_index = {}
        else:
            self.sessions_index = {}
    
    def _save_sessions_index(self):
        """保存会话索引"""
        try:
            with open(self.sessions_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.sessions_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存会话索引失败: {e}")
    
    def create_session(self, title: str = None, metadata: Dict[str, Any] = None) -> ChatSession:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        if not title:
            title = f"心血管咨询 - {now.strftime('%Y-%m-%d %H:%M')}"
        
        session = ChatSession(
            session_id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            messages=[],
            metadata=metadata or {}
        )
        
        # 更新索引
        self.sessions_index[session_id] = {
            'title': title,
            'created_at': now.isoformat(),
            'updated_at': now.isoformat(),
            'message_count': 0,
            'metadata': metadata or {}
        }
        
        self._save_sessions_index()
        self.current_session = session
        
        return session
    
    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """加载会话"""
        session_file = self.storage_dir / f"{session_id}.json"
        
        if not session_file.exists():
            print(f"会话文件不存在: {session_id}")
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session = ChatSession.from_dict(session_data)
            self.current_session = session
            return session
            
        except Exception as e:
            print(f"加载会话失败: {e}")
            return None
    
    def save_session(self, session: ChatSession = None):
        """保存会话"""
        if not session:
            session = self.current_session
        
        if not session:
            print("没有要保存的会话")
            return
        
        session_file = self.storage_dir / f"{session.session_id}.json"
        
        try:
            # 更新时间戳
            session.updated_at = datetime.now()
            
            # 保存会话文件
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 更新索引
            self.sessions_index[session.session_id] = {
                'title': session.title,
                'created_at': session.created_at.isoformat(),
                'updated_at': session.updated_at.isoformat(),
                'message_count': len(session.messages),
                'metadata': session.metadata or {}
            }
            
            self._save_sessions_index()
            
        except Exception as e:
            print(f"保存会话失败: {e}")
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> ChatMessage:
        """添加消息到当前会话"""
        if not self.current_session:
            self.create_session()
        
        message = ChatMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.current_session.messages.append(message)
        self.save_session()
        
        return message
    
    def get_sessions_list(self) -> List[Dict[str, Any]]:
        """获取会话列表"""
        sessions = []
        for session_id, info in self.sessions_index.items():
            sessions.append({
                'session_id': session_id,
                'title': info['title'],
                'created_at': info['created_at'],
                'updated_at': info['updated_at'],
                'message_count': info['message_count'],
                'metadata': info.get('metadata', {})
            })
        
        # 按更新时间排序
        sessions.sort(key=lambda x: x['updated_at'], reverse=True)
        return sessions
    
    def search_sessions(self, query: str) -> List[Dict[str, Any]]:
        """搜索会话"""
        results = []
        query_lower = query.lower()
        
        for session_id, info in self.sessions_index.items():
            # 搜索标题
            if query_lower in info['title'].lower():
                results.append({
                    'session_id': session_id,
                    'title': info['title'],
                    'match_type': 'title',
                    'created_at': info['created_at'],
                    'updated_at': info['updated_at'],
                    'message_count': info['message_count']
                })
                continue
            
            # 搜索消息内容
            session = self.load_session(session_id)
            if session:
                for message in session.messages:
                    if query_lower in message.content.lower():
                        results.append({
                            'session_id': session_id,
                            'title': info['title'],
                            'match_type': 'content',
                            'match_content': message.content[:100] + "...",
                            'created_at': info['created_at'],
                            'updated_at': info['updated_at'],
                            'message_count': info['message_count']
                        })
                        break
        
        return results
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        try:
            # 删除会话文件
            session_file = self.storage_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            # 从索引中删除
            if session_id in self.sessions_index:
                del self.sessions_index[session_id]
                self._save_sessions_index()
            
            # 如果是当前会话，清空
            if self.current_session and self.current_session.session_id == session_id:
                self.current_session = None
            
            return True
            
        except Exception as e:
            print(f"删除会话失败: {e}")
            return False
    
    def export_session(self, session_id: str, format: str = 'json') -> Optional[str]:
        """导出会话"""
        session = self.load_session(session_id)
        if not session:
            return None
        
        if format == 'json':
            return json.dumps(session.to_dict(), ensure_ascii=False, indent=2)
        
        elif format == 'txt':
            lines = [
                f"会话标题: {session.title}",
                f"创建时间: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                f"更新时间: {session.updated_at.strftime('%Y-%m-%d %H:%M:%S')}",
                f"消息数量: {len(session.messages)}",
                "=" * 50,
                ""
            ]
            
            for i, message in enumerate(session.messages, 1):
                role_name = {"user": "用户", "assistant": "助手", "system": "系统"}.get(message.role, message.role)
                lines.append(f"[{i}] {role_name} ({message.timestamp.strftime('%H:%M:%S')})")
                lines.append(message.content)
                lines.append("")
            
            return "\n".join(lines)
        
        return None
    
    def get_session_stats(self) -> Dict[str, Any]:
        """获取会话统计信息"""
        total_sessions = len(self.sessions_index)
        total_messages = sum(info['message_count'] for info in self.sessions_index.values())
        
        if total_sessions == 0:
            return {
                'total_sessions': 0,
                'total_messages': 0,
                'avg_messages_per_session': 0,
                'latest_session': None
            }
        
        # 最新会话
        latest_session = max(
            self.sessions_index.items(),
            key=lambda x: x[1]['updated_at']
        )
        
        return {
            'total_sessions': total_sessions,
            'total_messages': total_messages,
            'avg_messages_per_session': total_messages / total_sessions,
            'latest_session': {
                'session_id': latest_session[0],
                'title': latest_session[1]['title'],
                'updated_at': latest_session[1]['updated_at']
            }
        }
