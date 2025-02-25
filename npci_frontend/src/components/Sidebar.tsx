import React from 'react';
import { PlusCircle, Search, MessageSquare, Trash2, X } from 'lucide-react';
import { Conversation } from '../types';
import { format } from 'date-fns';

interface SidebarProps {
  conversations: Conversation[];
  activeConversation?: string;
  onNewChat: () => void;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onClose: () => void;
}

export function Sidebar({
  conversations,
  activeConversation,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  onClose,
}: SidebarProps) {
  const [searchTerm, setSearchTerm] = React.useState('');

  const filteredConversations = conversations.filter((conv) =>
    conv.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="h-full flex flex-col">
      <div className="p-2 border-b md:hidden">
        <button
          onClick={onClose}
          className="p-2 hover:bg-gray-100 rounded-lg"
        >
          <X className="w-6 h-6" />
        </button>
      </div>
      <div className="hidden md:flex w-80 bg-gray-50 border-r border-gray-200 h-screen flex-col">
      <div className="p-4">
          <button
            onClick={onNewChat}
            className="w-full flex items-center justify-center gap-2 bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
          >
            <PlusCircle size={20} />
            New Chat
          </button>
          
          <div className="mt-2 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              placeholder="Search conversations..."
              className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {filteredConversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`p-4 cursor-pointer hover:bg-gray-100 ${
                activeConversation === conversation.id ? 'bg-gray-100' : ''
              }`}
              onClick={() => onSelectConversation(conversation.id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <MessageSquare size={20} className="text-gray-500" />
                  <div>
                    <h3 className="font-medium text-gray-900">{conversation.title}</h3>
                    <p className="text-sm text-gray-500 mt-1">
                      {format(conversation.timestamp, 'MMM d, yyyy')}
                    </p>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteConversation(conversation.id);
                  }}
                  className="text-gray-400 hover:text-red-500 transition-colors"
                >
                  <Trash2 size={18} />
                </button>
              </div>
              {conversation.lastMessage && (
                <p className="text-sm text-gray-500 mt-2 truncate">
                  {conversation.lastMessage}
                </p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}