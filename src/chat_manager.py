"""
对话管理模块
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import uuid

from config import settings


@dataclass
class ChatMessage:
    """聊天消息"""
    id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatSession:
    """聊天会话"""
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[ChatMessage]
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'title': self.title,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'messages': [msg.to_dict() for msg in self.messages],
            'metadata': self.metadata or {}
        }


class ChatManager:
    """聊天管理器"""

    def __init__(self):
        self.current_session: Optional[ChatSession] = None
        self.sessions_dir = settings.CHAT_HISTORY_DIR
        os.makedirs(self.sessions_dir, exist_ok=True)

    def create_session(self, title: str = None) -> ChatSession:
        """
        创建新的聊天会话

        Args:
            title: 会话标题

        Returns:
            新创建的会话
        """
        session_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        if not title:
            title = f"医学咨询 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        session = ChatSession(
            session_id=session_id,
            title=title,
            created_at=current_time,
            updated_at=current_time,
            messages=[],
            metadata={'domain': 'medical', 'source': 'huatuo_rag'}
        )

        self.current_session = session
        self.save_session(session)
        return session

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> ChatMessage:
        """
        添加消息到当前会话

        Args:
            role: 角色 ('user' 或 'assistant')
            content: 消息内容
            metadata: 元数据

        Returns:
            添加的消息
        """
        if not self.current_session:
            self.create_session()

        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        message = ChatMessage(
            id=message_id,
            role=role,
            content=content,
            timestamp=timestamp,
            metadata=metadata or {}
        )

        self.current_session.messages.append(message)
        self.current_session.updated_at = timestamp

        # 自动保存
        self.save_session(self.current_session)

        return message

    def save_session(self, session: ChatSession) -> None:
        """
        保存会话到文件

        Args:
            session: 要保存的会话
        """
        try:
            filepath = os.path.join(
                self.sessions_dir,
                f"{session.session_id}.json"
            )

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"保存会话失败: {e}")

    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """
        加载会话

        Args:
            session_id: 会话ID

        Returns:
            加载的会话，如果不存在则返回None
        """
        try:
            filepath = os.path.join(self.sessions_dir, f"{session_id}.json")

            if not os.path.exists(filepath):
                return None

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 重构消息对象
            messages = [
                ChatMessage(**msg_data) for msg_data in data['messages']
            ]

            session = ChatSession(
                session_id=data['session_id'],
                title=data['title'],
                created_at=data['created_at'],
                updated_at=data['updated_at'],
                messages=messages,
                metadata=data.get('metadata', {})
            )

            self.current_session = session
            return session

        except Exception as e:
            print(f"加载会话失败: {e}")
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有会话

        Returns:
            会话列表（仅包含基本信息）
        """
        sessions = []

        try:
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.sessions_dir, filename)

                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    session_info = {
                        'session_id': data['session_id'],
                        'title': data['title'],
                        'created_at': data['created_at'],
                        'updated_at': data['updated_at'],
                        'message_count': len(data['messages'])
                    }
                    sessions.append(session_info)

            # 按更新时间排序
            sessions.sort(key=lambda x: x['updated_at'], reverse=True)

        except Exception as e:
            print(f"列出会话失败: {e}")

        return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            是否删除成功
        """
        try:
            filepath = os.path.join(self.sessions_dir, f"{session_id}.json")

            if os.path.exists(filepath):
                os.remove(filepath)

                # 如果删除的是当前会话，清空当前会话
                if (self.current_session and
                    self.current_session.session_id == session_id):
                    self.current_session = None

                return True

            return False

        except Exception as e:
            print(f"删除会话失败: {e}")
            return False

    def get_session_messages(
        self,
        session_id: str = None,
        limit: int = None
    ) -> List[ChatMessage]:
        """
        获取会话消息

        Args:
            session_id: 会话ID，None表示当前会话
            limit: 消息数量限制

        Returns:
            消息列表
        """
        if session_id:
            session = self.load_session(session_id)
        else:
            session = self.current_session

        if not session:
            return []

        messages = session.messages
        if limit:
            messages = messages[-limit:]

        return messages

    def search_messages(
        self,
        query: str,
        session_id: str = None
    ) -> List[ChatMessage]:
        """
        搜索消息

        Args:
            query: 搜索关键词
            session_id: 会话ID，None表示搜索所有会话

        Returns:
            匹配的消息列表
        """
        matching_messages = []

        try:
            if session_id:
                # 搜索特定会话
                session = self.load_session(session_id)
                if session:
                    for msg in session.messages:
                        if query.lower() in msg.content.lower():
                            matching_messages.append(msg)
            else:
                # 搜索所有会话
                for filename in os.listdir(self.sessions_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(self.sessions_dir, filename)

                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        for msg_data in data['messages']:
                            if query.lower() in msg_data['content'].lower():
                                matching_messages.append(ChatMessage(**msg_data))

        except Exception as e:
            print(f"搜索消息失败: {e}")

        return matching_messages

    def export_session(self, session_id: str, format: str = 'json') -> str:
        """
        导出会话

        Args:
            session_id: 会话ID
            format: 导出格式 ('json' 或 'txt')

        Returns:
            导出的内容
        """
        session = self.load_session(session_id)
        if not session:
            return ""

        if format == 'json':
            return json.dumps(session.to_dict(), ensure_ascii=False, indent=2)

        elif format == 'txt':
            lines = [
                f"会话标题: {session.title}",
                f"创建时间: {session.created_at}",
                f"更新时间: {session.updated_at}",
                "=" * 50,
                ""
            ]

            for msg in session.messages:
                role_name = "用户" if msg.role == "user" else "助手"
                lines.append(f"{role_name} ({msg.timestamp}):")
                lines.append(msg.content)
                lines.append("")

            return "\n".join(lines)

        return ""
