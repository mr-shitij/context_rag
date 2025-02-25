import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatContainer } from './components/ChatContainer';
import { Conversation, Message } from './types';
import clsx from 'clsx';

// Replace the mockApi with your actual API integration
const api = {
  async sendMessage(message: string): Promise<Message> {
    try {
      const response = await fetch('http://127.0.0.1:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Add any additional headers your API requires
          // 'Authorization': 'Bearer YOUR_API_KEY',
        },
        body: JSON.stringify({ message }), // This matches our API's expected format
      });

      if (!response.ok) {
        throw new Error('API request failed');
      }

      const data = await response.json();
      
      return {
        id: Math.random().toString(), // You might want to use an ID from your API response
        content: data.response, // This matches our API's response format
        role: 'assistant',
        timestamp: new Date(),
      };
    } catch (error) {
      console.error('Error calling API:', error);
      throw error;
    }
  },
};

// Simulated API calls - replace with actual API integration
// const mockApi = {
//   async sendMessage(message: string): Promise<Message> {
//     await new Promise(resolve => setTimeout(resolve, 1000));
//     return {
//       id: Math.random().toString(),
//       content: `Response to: ${message}`,
//       role: 'assistant',
//       timestamp: new Date(),
//     };
//   },
// };

export function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<string>();
  const [isLoading, setIsLoading] = useState(false);

  const currentConversation = conversations.find(c => c.id === activeConversation);

  const handleNewChat = () => {
    const newConversation: Conversation = {
      id: Math.random().toString(),
      title: 'New Conversation',
      timestamp: new Date(),
      messages: [],
    };
    setConversations(prev => [...prev, newConversation]);
    setActiveConversation(newConversation.id);
  };

  const handleSendMessage = async (content: string) => {
    if (!activeConversation) return;

    const userMessage: Message = {
      id: Math.random().toString(),
      content,
      role: 'user',
      timestamp: new Date(),
    };

    setConversations(prev =>
      prev.map(conv =>
        conv.id === activeConversation
          ? {
              ...conv,
              messages: [...conv.messages, userMessage],
              lastMessage: content,
            }
          : conv
      )
    );

    setIsLoading(true);
    try {
      // const response = await mockApi.sendMessage(content);
      const response = await api.sendMessage(content);
      setConversations(prev =>
        prev.map(conv =>
          conv.id === activeConversation
            ? {
                ...conv,
                messages: [...conv.messages, response],
                lastMessage: response.content,
              }
            : conv
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteConversation = (id: string) => {
    setConversations(prev => prev.filter(conv => conv.id !== id));
    if (activeConversation === id) {
      setActiveConversation(undefined);
    }
  };

  return (
    <div className="flex h-[100dvh] overflow-hidden">
      {/* Sidebar - hidden by default on mobile */}
      <div className={clsx(
        'fixed inset-y-0 left-0 z-50 w-64 bg-white transform transition-transform duration-300 ease-in-out md:relative md:translate-x-0',
        isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
      )}>
        <Sidebar
          conversations={conversations}
          activeConversation={activeConversation}
          onNewChat={handleNewChat}
          onSelectConversation={setActiveConversation}
          onDeleteConversation={handleDeleteConversation}
          onClose={() => setIsSidebarOpen(false)}
        />
      </div>

      {/* Overlay for mobile */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <div className="flex-1 flex">
        <ChatContainer
          messages={currentConversation?.messages || []}
          isLoading={isLoading}
          onSendMessage={handleSendMessage}
          onToggleSidebar={() => setIsSidebarOpen(true)}
        />
      </div>
    </div>
  );
}

export default App;